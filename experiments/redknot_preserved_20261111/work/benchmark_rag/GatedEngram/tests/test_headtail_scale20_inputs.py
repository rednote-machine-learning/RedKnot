"""Portable CPU input-contract tests using explicit synthetic unit-test fixtures.

No tokenizer, model, external data, or experiment output is used. Fixed original
production pins are patched only to the temporary unit fixture's byte hashes.
The native manifest parser is represented by its narrow documented interface;
its own token/formatter tests remain in the upstream suite.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpu_pilot import headtail_scale20_inputs as p


def seal(value):
    return p.sealed({k:v for k,v in value.items() if k!='sha256'})


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')


def source_fixture():
    sources={dataset:[(i+1,dict(_id=f'{dataset}-fixture-{i}',context=f'{dataset} synthetic context {i}',
        input=f'{dataset} synthetic question {i}',answers=[f'gold {i}'])) for i in range(28)]
        for dataset in p.DATASETS}
    rows=[]
    for index in range(6):
        dataset=p.DATASETS[index%2];source_index=index//2;raw=sources[dataset][source_index][1]
        rows.append(dict(index=index,dataset=dataset,source_id=raw['_id'],question=raw['input'],
            context_segments=[dict(kind='primary_full',dataset=dataset,source_id=raw['_id'],
                source_index=source_index,context_sha256=p.text_hash(raw['context']))]))
    original=seal(dict(warmup=rows[:2],cohort=rows[2:],selection_plan=dict(
        excluded_context_sha256=[p.text_hash(sources['hotpotqa'][3][1]['context'])],
        excluded_question_sha256=[p.text_hash(sources['2wikimqa'][3][1]['input'])])))
    return sources,original


class SourceSelectionTests(unittest.TestCase):
    def test_prior_context_question_identity_and_reserved_primaries_excluded(self):
        sources,original=source_fixture();sources=copy.deepcopy(sources)
        sources['hotpotqa'][4][1]['_id']=original['warmup'][0]['source_id']
        sources['hotpotqa'][5][1]['input']=original['cohort'][0]['question']
        sources['hotpotqa'][6][1]['context']=sources['hotpotqa'][7][1]['context']
        plan=p.build_source_plan(sources,original)
        rejected={x['source_index']:x['reason'] for x in plan['datasets']['hotpotqa']['rejected']}
        self.assertEqual(rejected[3],'old_context')
        self.assertEqual(rejected[4],'old_source_identity')
        self.assertEqual(rejected[5],'old_primary_question')
        self.assertEqual(rejected[7],'duplicate_context')
        primaries={x['context_sha256'] for dataset in p.DATASETS for x in plan['datasets'][dataset]['selected']}
        exclusions=set(plan['excluded_context_sha256'])
        self.assertEqual(sorted(x['index'] for dataset in p.DATASETS for x in plan['datasets'][dataset]['selected']),list(range(6,22)))
        for dataset in p.DATASETS:
            info=plan['datasets'][dataset]
            self.assertEqual(len(info['selected']),8)
            self.assertFalse(set(info['donor_context_sha256'])&(exclusions|primaries))
            self.assertEqual(info['donor_source_indices'],sorted(info['donor_source_indices']))

    def test_selection_is_deterministic_and_independent_of_gold_answers(self):
        sources,original=source_fixture();before=p.build_source_plan(sources,original)
        self.assertEqual(before,p.build_source_plan(sources,original))
        changed=copy.deepcopy(sources)
        for rows in changed.values():
            for _,raw in rows:raw['answers']=['changed outcome: never used to select']
        after=p.build_source_plan(changed,original)
        for dataset in p.DATASETS:
            self.assertEqual([x['source_index'] for x in before['datasets'][dataset]['selected']],
                [x['source_index'] for x in after['datasets'][dataset]['selected']])
            self.assertEqual(before['datasets'][dataset]['donor_source_indices'],after['datasets'][dataset]['donor_source_indices'])
        self.assertNotEqual(before['sha256'],after['sha256'])  # Raw source provenance still binds gold.

    def test_bad_seal_and_insufficient_unique_sources_fail_closed(self):
        sources,original=source_fixture();bad=copy.deepcopy(original);bad['cohort'][0]['question']='tampered'
        with self.assertRaises(ValueError):p.build_source_plan(sources,bad)
        with self.assertRaises(ValueError):p.build_source_plan({d:sources[d][:5] for d in p.DATASETS},original)

    def test_input_protocol_locks_tokenizer_version_and_fresh_twenty_measurements(self):
        sources,original=source_fixture();original=seal(dict(original,tokenizer={'fixture':'official-tokenizer-pin'}))
        plan=p.build_source_plan(sources,original);extension_protocol=p.input_protocol(original,plan,'0.21.4')
        extension=seal(dict(selection_plan=plan));combined=p.combined_protocol(original,extension,extension_protocol)
        self.assertEqual(combined['tokenizers_version'],'0.21.4')
        self.assertTrue(combined['fresh_measurements_required_for_all_twenty'])
        self.assertFalse(combined['historical_measurement_reuse'])
        self.assertTrue(combined['original_four_inputs_exact_copy'])
        self.assertEqual(combined['evaluation_indices'],list(range(2,22)))

    def test_strict_json_and_model_import_boundary_are_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'invalid.json'
            for payload in ('{"x":1,"x":2}','{"x":NaN}'):
                path.write_text(payload)
                with self.assertRaises(ValueError):p.read(path)
            path.write_text('{}');link=Path(directory)/'alias.json';link.symlink_to(path)
            with self.assertRaises(ValueError):p.read(link)
        guard=p.BlockAccelerators()
        for name in ('torch','torch.cuda','deepinfer.engine','triton','cupy'):
            with self.assertRaises(RuntimeError):guard.find_spec(name)
        self.assertIsNone(guard.find_spec('tokenizers'))


class ManifestInterface:
    """Minimal native-parser interface; the new contract itself is never mocked."""
    @staticmethod
    def validate_manifest(manifest):
        rows=manifest['warmup']+manifest['cohort']
        if len({x['index'] for x in rows})!=len(rows):raise ValueError('duplicate fixture row index')
        for row in rows:
            if row['input_tokens']!=len(row['input_ids']):raise ValueError('fixture input length mismatch')

    @staticmethod
    def document_specs(row):
        return [dict(doc,token_ids=row['input_ids'][doc['input_start']:doc['input_end']]) for doc in row['documents']]


def make_row(index,dataset,source_index,raw,anchor=1):
    # Actual 8x32K integer shape, deliberately synthetic: no tokenizer claims.
    ids=[1000+index]
    documents=[]
    for ordinal in range(8):
        values=[2000+index*8+ordinal]*32768;start=len(ids);ids.extend(values)
        documents.append(dict(input_start=start,input_end=len(ids),token_ids_sha256=p.digest(values)))
    ids.append(3000+index)
    return dict(index=index,dataset=dataset,source_index=source_index,source_id=raw['_id'],
        source_row_sha256=p.digest(raw),sample_id=f'unit-fixture:{index}',question=raw['input'],
        references=raw['answers'][:],references_sha256=p.digest(raw['answers']),primary_context_truncated=False,
        split='heldout',input_ids=ids,input_ids_sha256=p.digest(ids),input_tokens=len(ids),documents=documents,
        context_tokens=262144,prefix_tokens=1,suffix_tokens=1,original_context_tokens=5,
        context_segments=[dict(kind='primary_full',dataset=dataset,source_id=raw['_id'],source_index=source_index,
            context_sha256=p.text_hash(raw['context']),context_start=anchor*32768,selected_tokens=5)])


class ExtensionAndAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        sources,skeleton=source_fixture();cls.sources=sources;metadata=[]
        for dataset,records in sources.items():
            path=cls.root/(dataset+'.jsonl')
            path.write_text(''.join(json.dumps(raw)+'\n' for _,raw in records))
            metadata.append(dict(dataset=dataset,path=str(path),sha256=p.file_hash(path)))
        rows=[make_row(i,p.DATASETS[i%2],i//2,sources[p.DATASETS[i%2]][i//2][1]) for i in range(6)]
        cls.original=seal(dict(skeleton,warmup=rows[:2],cohort=rows[2:],sources=metadata,
            tokenizer={'tokenizer_json_sha256':'a'*64,'encoding_sha256':'b'*64},instruction='synthetic unit fixture'))
        cls.plan=p.build_source_plan(sources,cls.original)
        cls.extension_protocol=p.input_protocol(cls.original,cls.plan,'0.21.4')
        rows=[]
        for ordinal in range(8):
            for dataset in p.DATASETS:
                selected=cls.plan['datasets'][dataset]['selected'][ordinal]
                source_index=selected['source_index'];raw=sources[dataset][source_index][1]
                rows.append(make_row(selected['index'],dataset,source_index,raw,selected['anchor_chunk']))
        cls.extension=seal(dict(artifact_kind='headtail_link_extension_inputs_v1',
            extension_input_protocol_sha256=cls.extension_protocol['sha256'],original_manifest_sha256=cls.original['sha256'],
            selection_plan=cls.plan,warmup=cls.original['warmup'],cohort=rows))
        cls.original_path=cls.root/'source-manifest.json';write(cls.original_path,cls.original)
        cls.extension_dir=cls.root/'extension';cls.extension_dir.mkdir()
        files={}
        for name,value in [('manifest.json',cls.extension),('input-protocol.json',cls.extension_protocol),('source-plan.json',cls.plan)]:
            path=cls.extension_dir/name;write(path,value)
            files[name]=dict(file_sha256=p.file_hash(path),seal_sha256=value['sha256'])
        cls.preparation=seal(dict(schema='headtail-link-extension-input-preparation-v1',status='complete',
            actual_token_inputs_generated=True,new_evaluation_cases=16,cuda_initialized=False,files=files))
        write(cls.extension_dir/'preparation.json',cls.preparation)
        cls.pin_file=p.file_hash(cls.original_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def validate(self,manifest):
        return p.validate_extension(manifest,self.extension_protocol,self.original,self.plan,self.sources,ManifestInterface)

    def mutate_row(self,update,index=0):
        rows=list(self.extension['cohort']);rows[index]=dict(rows[index],**update)
        return seal(dict(self.extension,cohort=rows))

    def pins(self):
        stack=contextlib.ExitStack()
        stack.enter_context(patch.object(p,'ORIGINAL_FILE',self.pin_file))
        stack.enter_context(patch.object(p,'ORIGINAL_SEAL',self.original['sha256']))
        return stack

    def test_real_new_contract_accepts_balanced_16_full_sized_fixture(self):
        self.assertTrue(self.validate(self.extension))
        self.assertEqual(len(self.extension['cohort']),16)

    def test_gold_calibration_copy_and_whole_primary_are_immutable(self):
        changes=[dict(references=['changed gold'],references_sha256=p.digest(['changed gold'])),
            dict(primary_context_truncated=True),dict(source_row_sha256='f'*64),dict(split='calibration')]
        for update in changes:
            with self.subTest(update=update),self.assertRaises(ValueError):self.validate(self.mutate_row(update))
        calibration=[dict(self.original['warmup'][0],references=['changed calibration']),self.original['warmup'][1]]
        with self.assertRaises(ValueError):self.validate(seal(dict(self.extension,warmup=calibration)))

    def test_source_segments_old_documents_and_shape_cannot_be_relabelled(self):
        current=self.extension['cohort'][0]
        docs=[dict(x) for x in current['documents']];docs[0]['token_ids_sha256']=self.original['warmup'][0]['documents'][0]['token_ids_sha256']
        with self.assertRaises(ValueError):self.validate(self.mutate_row(dict(documents=docs)))
        docs=[dict(x) for x in current['documents']];docs[0]['input_end']-=1
        with self.assertRaises(ValueError):self.validate(self.mutate_row(dict(documents=docs)))
        segment=dict(current['context_segments'][0]);segment['source_id']=self.original['warmup'][0]['source_id']
        with self.assertRaises(ValueError):self.validate(self.mutate_row(dict(context_segments=[segment])))
        donor=dict(current['context_segments'][0],kind='donor')
        with self.assertRaises(ValueError):self.validate(self.mutate_row(dict(context_segments=[current['context_segments'][0],donor])))

    def test_assembly_copies_originals_and_companion_bytes_and_validates_all_22(self):
        output=self.root/'assembled';args=SimpleNamespace(extension_dir=self.extension_dir,
            source_manifest=self.original_path,output_dir=output)
        with self.pins(),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(p.assemble_inputs(args,self.original,ManifestInterface),0)
            manifest=p.read(output/'manifest.json');protocol=p.read(output/'input-protocol.json')
            self.assertEqual(len(p.validate_inputs(ManifestInterface,manifest,protocol,input_dir=output)),22)
        self.assertEqual(manifest['warmup'],self.original['warmup'])
        self.assertEqual(manifest['cohort'][:4],self.original['cohort'])
        self.assertEqual(manifest['cohort'][4:],self.extension['cohort'])
        for name,source in [('source-manifest.json',self.original_path),('extension-manifest.json',self.extension_dir/'manifest.json'),
            ('extension-input-protocol.json',self.extension_dir/'input-protocol.json'),('source-plan.json',self.extension_dir/'source-plan.json')]:
            self.assertEqual((output/name).read_bytes(),source.read_bytes())
        preparation=p.read(output/'preparation.json');p.verify_seal(preparation)
        self.assertFalse(preparation['model_results_generated'])
        self.assertFalse(preparation['historical_measurement_reuse'])
        with self.pins(),self.assertRaises(FileExistsError),contextlib.redirect_stdout(io.StringIO()):
            p.assemble_inputs(args,self.original,ManifestInterface)

    def test_incomplete_or_tampered_preparation_never_creates_combined_inputs(self):
        before=(self.extension_dir/'preparation.json').read_bytes()
        try:
            for update in (dict(actual_token_inputs_generated=False),dict(cuda_initialized=True),dict(status='preparing')):
                write(self.extension_dir/'preparation.json',seal(dict(self.preparation,**update)))
                output=self.root/'rejected';args=SimpleNamespace(extension_dir=self.extension_dir,
                    source_manifest=self.original_path,output_dir=output)
                with self.pins(),self.assertRaises(ValueError):p.assemble_inputs(args,self.original,ManifestInterface)
                self.assertFalse(output.exists())
            changed=copy.deepcopy(self.preparation);changed['files']['manifest.json']['file_sha256']='f'*64
            write(self.extension_dir/'preparation.json',seal(changed))
            with self.pins(),self.assertRaises(ValueError):p.assemble_inputs(args,self.original,ManifestInterface)
            self.assertFalse(output.exists())
        finally:(self.extension_dir/'preparation.json').write_bytes(before)

    def test_combined_validator_rejects_original_mutation_and_companion_substitution(self):
        # Build an independent fixture directory without bypassing either validator.
        output=self.root/'validation';args=SimpleNamespace(extension_dir=self.extension_dir,
            source_manifest=self.original_path,output_dir=output)
        with self.pins(),contextlib.redirect_stdout(io.StringIO()):p.assemble_inputs(args,self.original,ManifestInterface)
        manifest=p.read(output/'manifest.json');protocol=p.read(output/'input-protocol.json')
        rows=list(manifest['cohort']);rows[0]=dict(rows[0],question='rewritten original question')
        with self.pins(),self.assertRaises(ValueError):
            p.validate_inputs(ManifestInterface,seal(dict(manifest,cohort=rows)),protocol,input_dir=output)
        path=output/'source-plan.json';bad=dict(self.plan,seed=17);write(path,seal(bad))
        with self.pins(),self.assertRaises(ValueError):p.validate_inputs(ManifestInterface,manifest,protocol,input_dir=output)
        write(path,self.plan)
        path=output/'source-manifest.json';path.write_bytes(path.read_bytes()+b' ')
        with self.pins(),self.assertRaises(ValueError):p.validate_inputs(ManifestInterface,manifest,protocol,input_dir=output)


if __name__=='__main__':unittest.main()
