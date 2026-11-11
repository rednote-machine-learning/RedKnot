"""Independent combinatorial partition oracle and resealed input mutation tests."""
import copy
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import natural_boundary_inputs as n
from tests.test_natural_boundary_inputs import CharacterTokenizer


def renderer(messages,thinking_mode):
    assert thinking_mode=='chat'
    return '<user>'+messages[0]['content']+'</user><assistant>'


def fixture():
    templates={dataset:'Passages:\n{context}\nQuestion: {input}\nAnswer:' for dataset in n.DATASETS}
    template_content=json.dumps(templates,sort_keys=True)
    identity=dict(encoding_sha256=n.OFFICIAL_ENCODING_SHA256,tokenizer_json_sha256='c'*64,
        official_longbench_prompt_sha256=n.text_hash(template_content),tokenizers_version='0.21.4')
    rows=[];source_records=[];token_records=[]
    for index in range(22):
        dataset=n.DATASETS[index%2]
        raw=dict(_id='sample'+str(index),input=f'Which original entity for sample {index}?',answers=[f'Entity {index}'],
            context='\n\n'.join(f'Passage {p+1}:\nSource {index} '+chr(97+p)*2050 for p in range(8)))
        row,evidence=n.build_row(raw,dataset,index,CharacterTokenizer(),renderer,templates[dataset])
        row.update(index=index,split='calibration' if index<2 else 'evaluation')
        rows.append(row);token_records.append(evidence)
        source_records.append(dict(sample_id=row['sample_id'],dataset=dataset,source_index=index,
            source_row_sha256=n.digest(raw),raw=raw))
    companions={
        'source-records.json':dict(schema='natural-boundary-source-records-v1',records=source_records),
        'selection-plan.json':dict(schema='natural-boundary-selection-plan-v1',seed=n.SEED,no_model_outcomes_used=True,
            selected_sample_ids=[r['sample_id'] for r in rows]),
        'tokenization-evidence.json':dict(schema='natural-boundary-tokenization-evidence-v1',tokenizer=identity,
            official_prompt_file_content=template_content,official_prompt_source=dict(sha256=n.text_hash(template_content)),
            official_prompt_templates=templates,official_encoding_source='unit-test-renderer',records=token_records)}
    protocol=dict(schema=n.INPUT_PROTOCOL_SCHEMA,composition='single_original_qa_eight_natural_cache_blocks',
        no_padding=True,no_truncation=True,no_cross_sample_contexts=True,preserve_original_order=True)
    manifest=dict(schema=n.MANIFEST_SCHEMA,contract_schema=n.CONTRACT_SCHEMA,artifact_kind='natural_boundary_inputs_v1',
        tokenizer=identity,warmup=rows[:2],cohort=rows[2:])
    return manifest,protocol,companions


def reseal_and_validate(values):
    manifest,protocol,companions=copy.deepcopy(values)
    for name,value in companions.items():
        companions[name]=n.seal({k:v for k,v in value.items() if k!='sha256'})
    protocol.update(source_records_sha256=companions['source-records.json']['sha256'],
        selection_plan_sha256=companions['selection-plan.json']['sha256'],
        tokenization_evidence_sha256=companions['tokenization-evidence.json']['sha256'])
    protocol=n.seal({k:v for k,v in protocol.items() if k!='sha256'})
    manifest.update(companion_sha256={name:value['sha256'] for name,value in companions.items()},
        input_protocol_sha256=protocol['sha256'])
    manifest=n.seal({k:v for k,v in manifest.items() if k!='sha256'})
    with tempfile.TemporaryDirectory() as directory:
        for name,value in companions.items():
            (Path(directory)/name).write_text(json.dumps(value,ensure_ascii=False))
        # This fixture exercises contract validation; production renderer SHA is
        # checked separately by frozen_renderer and never patched in execution.
        with patch.object(n,'frozen_renderer',return_value=renderer):
            return n.validate_inputs(None,manifest,protocol,input_dir=directory)


class NaturalBoundaryIndependentInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.good=fixture()

    def test_complete_twenty_plus_two_contract_accepts_only_bound_companions(self):
        self.assertTrue(reseal_and_validate(self.good))
        bad=copy.deepcopy(self.good);bad[2]['selection-plan.json']['selected_sample_ids'].reverse()
        with self.assertRaises(ValueError):reseal_and_validate(bad)
        bad=copy.deepcopy(self.good);bad[2]['selection-plan.json']['no_model_outcomes_used']=False
        with self.assertRaises(ValueError):reseal_and_validate(bad)

    def test_source_gold_and_extra_prompt_injection_rejected_after_resealing(self):
        bad=copy.deepcopy(self.good)
        bad[0]['cohort'][0]['references']=['changed gold']
        bad[0]['cohort'][0]['references_sha256']=n.digest(['changed gold'])
        with self.assertRaises(ValueError):reseal_and_validate(bad)
        bad=copy.deepcopy(self.good);row=bad[0]['cohort'][0]
        row['prompt']+='\nQuestion: another sample?\nAnswer: injected'
        row['prompt_sha256']=n.text_hash(row['prompt'])
        bad[2]['tokenization-evidence.json']['records'][2]['prompt_sha256']=row['prompt_sha256']
        with self.assertRaisesRegex(ValueError,'exact original QA'):reseal_and_validate(bad)
        bad=copy.deepcopy(self.good);raw=bad[2]['source-records.json']['records'][2]
        raw['raw']['context']+=' additional source text';raw['source_row_sha256']=n.digest(raw['raw'])
        with self.assertRaises(ValueError):reseal_and_validate(bad)

    def test_character_boundary_and_offset_mutations_rejected_even_with_valid_seals(self):
        bad=copy.deepcopy(self.good);bad[0]['cohort'][0]['documents'][0]['context_char_end']+=1
        with self.assertRaises(ValueError):reseal_and_validate(bad)
        bad=copy.deepcopy(self.good);bad[2]['tokenization-evidence.json']['records'][2]['token_offsets'][0]=[99,100]
        with self.assertRaisesRegex(ValueError,'monotonic'):reseal_and_validate(bad)
        bad=copy.deepcopy(self.good);bad[2]['tokenization-evidence.json']['records'][2]['whole_prompt_single_encoding']=False
        with self.assertRaises(ValueError):reseal_and_validate(bad)

    def test_pinned_renderer_source_cannot_be_swapped(self):
        with self.assertRaisesRegex(ValueError,'unrecognized official'):
            n.frozen_renderer('def encode_messages(*args,**kwargs): return "fake"')

    def test_partition_global_optimum_and_tie_break_match_exhaustive_enumeration(self):
        rng=random.Random(1837)
        for lengths in ([300]*10, *[[rng.randrange(160,700) for _ in range(10)] for _ in range(12)]):
            boundaries=[0]
            for length in lengths:boundaries.append(boundaries[-1]+length)
            offsets=[(i,i+1) for i in range(boundaries[-1])]
            total=boundaries[-1];candidates=[]
            for cuts in itertools.combinations(range(1,len(lengths)),7):
                stops=(0,*cuts,len(lengths))
                sizes=[boundaries[b]-boundaries[a] for a,b in zip(stops,stops[1:])]
                if min(sizes)>=256:
                    candidates.append((sum((8*size-total)**2 for size in sizes),(*cuts,len(lengths))))
            if not candidates:
                with self.assertRaises(ValueError):n.partition_eight(boundaries,offsets,0)
                continue
            expected=min(candidates)[1]
            actual=n.partition_eight(boundaries,offsets,0)
            self.assertEqual(tuple(group['passage_end'] for group in actual),expected)
            self.assertEqual(actual[0]['char_start'],0);self.assertEqual(actual[-1]['char_end'],total)

    def test_tokens_crossing_cut_are_online_once_and_never_duplicated_between_blocks(self):
        offsets=[(0,2),(2,6),(6,9),(9,12)]
        left=n.exact_token_span(offsets,0,4);right=n.exact_token_span(offsets,4,12)
        cached=set(range(*left))|set(range(*right))
        self.assertEqual(set(range(*left))&set(range(*right)),set())
        self.assertEqual(set(range(4))-cached,{1})
        self.assertEqual(offsets[1],(2,6))


if __name__=='__main__':unittest.main()
