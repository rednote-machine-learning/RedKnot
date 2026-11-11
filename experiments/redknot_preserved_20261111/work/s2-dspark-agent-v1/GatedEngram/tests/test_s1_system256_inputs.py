"""Sealed-probe conversion and mutation rejection, no tokenizer or GPU calls."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gpu_pilot import s1_system256_inputs as m
from tests import test_loong_inputs as fixtures


class System256Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.LoongInputs.setUpClass()
        cls.previous={name:copy.deepcopy(fixtures.LoongInputs.artifacts[name]) for name in m.PREVIOUS_FILES}
        selected=[copy.deepcopy(r) for r in cls.previous['manifest.json']['cohort'] if r['length_band']=='128k'][:3]
        for i,row in enumerate(selected,1):
            for k in ('index','split','length_band_ordinal'):row.pop(k)
            row.update(selection_ordinal=i,length_band='256k',target_input_tokens=9000,
                technical_max_new_tokens=m.ENGINE_MAX_MODEL_LEN-row['input_tokens'])
        ids={r['sample_id'] for r in selected};files={d['source_file'] for r in selected for d in r['original_documents']}
        source={k:copy.deepcopy(v) for k,v in cls.previous['source-records.json'].items() if k not in ('sha256','previous_inputs')}
        source['schema']='loong256k-source-records-v1'
        source['records']=[{k:v for k,v in r.items() if k!='dataset'} for r in source['records'] if r['sample_id'] in ids]
        source['document_files']={k:v for k,v in source['document_files'].items() if k in files}
        tokens={k:copy.deepcopy(v) for k,v in cls.previous['tokenization-evidence.json'].items() if k!='sha256'}
        tokens['schema']='loong256k-tokenization-evidence-v1';tokens['records']=[r for r in tokens['records'] if r['sample_id'] in ids]
        # Actual probe has six execution fields; old calibration retains two
        # additional historical LongBench prompt provenance fields.
        tokens['tokenizer']={k:v for k,v in tokens['tokenizer'].items() if k not in m.CALIBRATION_PROMPT_PROVENANCE_FIELDS}
        source,tokens=m.seal(source),m.seal(tokens)
        selection=m.seal(dict(schema='loong256k-selection-plan-v1',selected_sample_ids=[r['sample_id'] for r in selected],
            selected_count=3,requested_count=3,no_model_outcomes_used=True,no_answer_content_selection=True))
        chosen=m.seal(dict(schema='loong256k-selected-rows-v1',formal_experiment_manifest=False,calibration_rows_added=False,
            source_records_sha256=source['sha256'],selection_plan_sha256=selection['sha256'],tokenization_evidence_sha256=tokens['sha256'],rows=selected))
        prep=m.seal(dict(schema='loong256k-preparation-v1',status='complete',selected_count=3,requested_count=3,
            cuda_used=False,inference_run=False,model_outputs_used=False,selected_rows_sha256=chosen['sha256'],
            actual_tokenizer_identity=tokens['tokenizer']))
        cls.probe=dict(zip(m.INPUT_COMPANIONS,(source,selection,tokens)))
        cls.probe.update({'selected-rows.json':chosen,'preparation.json':prep})
        with cls.pins():cls.artifacts=m.build_inputs(cls.previous,cls.probe)

    @classmethod
    def pins(cls):
        stack=contextlib.ExitStack();stack.enter_context(fixtures.LoongInputs.pins())
        values=dict(PREVIOUS_MANIFEST_SHA256=cls.previous['manifest.json']['sha256'],
            PREVIOUS_PROTOCOL_SHA256=cls.previous['input-protocol.json']['sha256'],
            PROBE_PREPARATION_SHA256=cls.probe['preparation.json']['sha256'],
            PROBE_SELECTED_ROWS_SHA256=cls.probe['selected-rows.json']['sha256'],
            SELECTED_SOURCE_INDICES=tuple(r['source_index'] for r in cls.probe['selected-rows.json']['rows']),
            INPUT_TOKEN_RANGE=(8000,11000),TARGET_INPUT_TOKENS=9000)
        for k,v in values.items():stack.enter_context(patch.object(m,k,v))
        return stack

    def check(self,data=None):
        data=data or self.artifacts
        with self.pins(),tempfile.TemporaryDirectory() as temp:
            for name in m.INPUT_COMPANIONS:Path(temp,name).write_text(json.dumps(data[name]))
            return m.validate_inputs(None,data['manifest.json'],data['input-protocol.json'],input_dir=temp)

    def rebind(self,data):
        for name in m.INPUT_COMPANIONS:data[name]=m.seal({k:v for k,v in data[name].items() if k!='sha256'})
        for key,name in zip(('source_records_sha256','selection_plan_sha256','tokenization_evidence_sha256'),m.INPUT_COMPANIONS):
            data['input-protocol.json'][key]=data[name]['sha256']
        data['input-protocol.json']=m.seal({k:v for k,v in data['input-protocol.json'].items() if k!='sha256'})
        manifest=data['manifest.json'];manifest['input_protocol_sha256']=data['input-protocol.json']['sha256']
        manifest['companion_sha256']={n:data[n]['sha256'] for n in m.INPUT_COMPANIONS}
        manifest['selection_plan_sha256']=data['selection-plan.json']['sha256']
        data['manifest.json']=m.seal({k:v for k,v in manifest.items() if k!='sha256'})

    def test_exact_three_rows_and_calibration_without_retokenization(self):
        self.assertTrue(self.check());v=self.artifacts['manifest.json']
        self.assertEqual(v['warmup'],self.previous['manifest.json']['warmup'])
        self.assertEqual([r['index'] for r in v['cohort']],[2,3,4])
        for actual,original in zip(v['cohort'],self.probe['selected-rows.json']['rows']):
            self.assertEqual({k:v for k,v in actual.items() if k not in ('index','split','length_band_ordinal')},original)
        self.assertFalse(self.artifacts['preparation.json']['new_tokenization_executed'])

    def test_original_probe_cannot_be_resealed_to_change_gold_or_selection(self):
        for key,value in (('raw_answer','changed'),('input_tokens',999),('source_index',22)):
            bad=copy.deepcopy(self.probe);bad['selected-rows.json']['rows'][0][key]=value
            bad['selected-rows.json']=m.seal({k:v for k,v in bad['selected-rows.json'].items() if k!='sha256'})
            with self.pins(),self.assertRaises(ValueError):m.build_inputs(self.previous,bad)

    def test_resealed_formal_rows_cannot_rewrite_original_tokens_gold_or_geometry(self):
        for field in ('input_ids','raw_answer','question','documents','technical_max_new_tokens','length_band'):
            bad=copy.deepcopy(self.artifacts);r=bad['manifest.json']['cohort'][0]
            if field=='input_ids':r[field][0]+=1
            elif field=='documents':r[field][0]['input_end']-=1
            elif field=='technical_max_new_tokens':r[field]+=1
            else:r[field]='changed'
            self.rebind(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(bad)

    def test_resealed_source_or_offsets_cannot_escape_original_probe_seal(self):
        for kind in ('source','offsets','source_record','selection'):
            bad=copy.deepcopy(self.artifacts)
            if kind=='source':
                files=bad['source-records.json']['document_files'];files[next(iter(files))]+=' filler'
            elif kind=='offsets':bad['tokenization-evidence.json']['records'][-1]['token_offsets'].pop()
            elif kind=='source_record':bad['source-records.json']['records'][-1]['raw']['answer']='changed'
            else:bad['selection-plan.json']['selected_sample_ids'].reverse()
            self.rebind(bad)
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.check(bad)

    def test_calibration_is_exact_original128_and_native_ancestry(self):
        bad=copy.deepcopy(self.artifacts);bad['manifest.json']['warmup'][0]['input_ids'][0]+=1;self.rebind(bad)
        with self.assertRaises(ValueError):self.check(bad)
        old=copy.deepcopy(self.previous);old['manifest.json']['warmup'][0]['question']='changed'
        old['manifest.json']=m.seal({k:v for k,v in old['manifest.json'].items() if k!='sha256'})
        with self.pins(),self.assertRaises(ValueError):m.build_inputs(old,self.probe)

    def test_missing_or_extra_evaluation_row_and_reordering_are_refused(self):
        for op in ('missing','extra','reverse'):
            bad=copy.deepcopy(self.artifacts);rows=bad['manifest.json']['cohort']
            if op=='missing':rows.pop()
            elif op=='extra':rows.append(copy.deepcopy(rows[-1]))
            else:rows.reverse()
            self.rebind(bad)
            with self.assertRaises(ValueError):self.check(bad)

    def test_formal_minimum_output_headroom_and_core_block_count(self):
        self.assertEqual(m.ENGINE_MAX_MODEL_LEN-m.INPUT_TOKEN_RANGE[1],4096)
        policy=m.fixed_policy();self.assertEqual(policy['num_chunks'],10)
        self.assertFalse(policy['reselection_performed']);self.assertTrue(policy['natural_eos_preserved'])

    def test_actual_six_field_probe_and_eight_field_calibration_both_preserved(self):
        probe=self.probe['tokenization-evidence.json']['tokenizer'];prior=self.previous['manifest.json']['tokenizer']
        self.assertEqual(set(probe),set(m.TOKENIZER_EXECUTION_FIELDS))
        self.assertEqual(set(prior),set(probe)|set(m.CALIBRATION_PROMPT_PROVENANCE_FIELDS))
        self.assertEqual(self.artifacts['tokenization-evidence.json']['tokenizer'],probe)
        self.assertEqual(self.artifacts['manifest.json']['tokenizer'],prior)
        protocol=self.artifacts['input-protocol.json']
        self.assertEqual(protocol['actual_tokenizer_identity'],probe)
        self.assertEqual(protocol['calibration_prompt_provenance'],{k:prior[k] for k in m.CALIBRATION_PROMPT_PROVENANCE_FIELDS})
        self.assertTrue(self.check())  # Reconstructs and verifies original six-field probe seals.

    def test_every_actual_tokenizer_identity_mismatch_refused_even_with_probe_pins(self):
        for key in m.TOKENIZER_EXECUTION_FIELDS:
            bad=copy.deepcopy(self.probe);token=bad['tokenization-evidence.json'];token['tokenizer'][key]+='-changed'
            bad['tokenization-evidence.json']=m.seal({k:v for k,v in token.items() if k!='sha256'})
            chosen=bad['selected-rows.json'];chosen['tokenization_evidence_sha256']=bad['tokenization-evidence.json']['sha256']
            bad['selected-rows.json']=m.seal({k:v for k,v in chosen.items() if k!='sha256'})
            prep=bad['preparation.json'];prep['actual_tokenizer_identity']=token['tokenizer'];prep['selected_rows_sha256']=bad['selected-rows.json']['sha256']
            bad['preparation.json']=m.seal({k:v for k,v in prep.items() if k!='sha256'})
            with self.subTest(key=key),self.pins(),patch.object(m,'PROBE_SELECTED_ROWS_SHA256',bad['selected-rows.json']['sha256']),\
                patch.object(m,'PROBE_PREPARATION_SHA256',bad['preparation.json']['sha256']),\
                self.assertRaisesRegex(ValueError,'actual tokenizer differ'):
                m.build_inputs(self.previous,bad)

    def test_unknown_metadata_or_rewritten_calibration_provenance_refused(self):
        value=copy.deepcopy(self.probe['tokenization-evidence.json']['tokenizer']);value['unbound_execution_option']='on'
        with self.assertRaises(ValueError):m.tokenizer_execution_identity(value)
        value.pop('unbound_execution_option');value['official_longbench_prompt_revision']='partial'
        with self.assertRaises(ValueError):m.tokenizer_execution_identity(value)
        bad=copy.deepcopy(self.artifacts);bad['input-protocol.json']['calibration_prompt_provenance']['official_longbench_prompt_revision']='changed';self.rebind(bad)
        with self.assertRaisesRegex(ValueError,'prompt provenance differs'):self.check(bad)


if __name__=='__main__':unittest.main()
