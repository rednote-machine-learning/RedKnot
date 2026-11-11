"""Real producer bundle composition; no model outputs, tokenizer or GPU used."""
import contextlib
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpu_pilot import s11_inputs as m
from tests import test_s1_system256_inputs as fixtures


class S11Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.System256Inputs.setUpClass()
        cls.original128 = fixtures.System256Inputs.previous
        cls.system256 = {n: fixtures.System256Inputs.artifacts[n] for n in m.ORIGINAL_FILES}
        ref = lambda name: dict(path='/evidence/'+name+'.json', file_sha256=m.digest(name), seal_sha256=m.digest([name]))
        cls.predecessor = m.seal(dict(schema=m.PREDECESSOR_SCHEMA, status='accepted_user_stop',
            manifest_sha256=cls.original128['manifest.json']['sha256'],
            registered_case_indices=list(range(2,22,2)), completed_case_indices=list(range(2,20,2)),
            pending_case_indices=[20], new_evaluation_case_indices=[20,2,3,4],
            source_run_count=2, single_run=False, exposed_arms=['recomputed','s1_system'],
            user_requested_stop_authenticated=True, physical_idle_verified=True, stable_scope={'fixture':'CPU only'},
            user_stop_request=ref('stop'), physical_idle=ref('idle'),
            audits=[dict(run_id=name, accepted_case_indices=indices,
                **{k:ref(name+'-'+k) for k in ('audit','preflight','workflow_terminal','lifecycle_terminal')})
                for name,indices in (('first',[2,4,6]),('continuation',[8,10,12,14,16,18]))]))
        with cls.pins():
            cls.artifacts = m.build_inputs(cls.original128, cls.system256, cls.predecessor,
                                          expected_predecessor_sha256=cls.predecessor['sha256'])

    @classmethod
    def pins(cls):
        stack = contextlib.ExitStack(); stack.enter_context(fixtures.System256Inputs.pins())
        for key,value in (('SYSTEM256_MANIFEST_SHA256',cls.system256['manifest.json']['sha256']),
                          ('SYSTEM256_PROTOCOL_SHA256',cls.system256['input-protocol.json']['sha256'])):
            stack.enter_context(patch.object(m,key,value))
        return stack

    def check(self, data=None, base=None):
        data = self.artifacts if data is None else data
        with self.pins():
            return m._validate_bundle(base,data['manifest.json'],data['input-protocol.json'],
                                      {n:data[n] for n in m.INPUT_COMPANIONS})

    @staticmethod
    def reseal(data):
        for name in m.INPUT_COMPANIONS: data[name] = m.seal({k:v for k,v in data[name].items() if k!='sha256'})
        protocol = data['input-protocol.json']
        for key,name in zip(('source_records_sha256','selection_plan_sha256','tokenization_evidence_sha256'),m.INPUT_COMPANIONS):
            protocol[key] = data[name]['sha256']
        data['input-protocol.json'] = m.seal({k:v for k,v in protocol.items() if k!='sha256'})
        manifest=data['manifest.json'];manifest['input_protocol_sha256']=data['input-protocol.json']['sha256']
        manifest['companion_sha256']={n:data[n]['sha256'] for n in m.INPUT_COMPANIONS}
        manifest['selection_plan_sha256']=data['selection-plan.json']['sha256']
        data['manifest.json']=m.seal({k:v for k,v in manifest.items() if k!='sha256'})

    def test_actual_original_producers_combine_six_unchanged_rows(self):
        self.assertTrue(self.check())
        manifest=self.artifacts['manifest.json'];protocol=self.artifacts['input-protocol.json']
        self.assertEqual(manifest['warmup'],self.system256['manifest.json']['warmup'])
        self.assertEqual(manifest['cohort'][:3],self.system256['manifest.json']['cohort'])
        self.assertEqual(manifest['cohort'][3],next(r for r in self.original128['manifest.json']['cohort'] if r['index']==20))
        self.assertEqual(protocol['execution_indices'],[20,2,3,4]);self.assertEqual(protocol['qualification_anchor_index'],2)
        self.assertEqual([r['index'] for r in manifest['warmup']+manifest['cohort']],[0,1,2,3,4,20])
        self.assertFalse(self.artifacts['preparation.json']['new_tokenization_executed'])
        self.assertEqual(protocol['source_quality_annotations'],self.system256['input-protocol.json']['source_quality_annotations'])

    def test_remaining_selection_comes_only_from_pinned_completed_population(self):
        for field,value in (('pending_case_indices',[18,20]),('completed_case_indices',list(range(2,18,2))),
                            ('registered_case_indices',[2,4,20]),('physical_idle_verified',False),
                            ('user_requested_stop_authenticated',False),('manifest_sha256','a'*64)):
            receipt=dict(self.predecessor);receipt[field]=value
            receipt=m.seal({k:v for k,v in receipt.items() if k!='sha256'})
            with self.subTest(field=field),self.pins(),self.assertRaises(ValueError):
                m.build_inputs(self.original128,self.system256,receipt,expected_predecessor_sha256=receipt['sha256'])

    def test_receipt_external_pin_and_disjoint_audit_partitions_required(self):
        with self.pins(),self.assertRaisesRegex(ValueError,'pinned predecessor'):
            m.build_inputs(self.original128,self.system256,self.predecessor,expected_predecessor_sha256='a'*64)
        bad=copy.deepcopy(self.predecessor);bad['audits'][1]['accepted_case_indices'][0]=6
        bad=m.seal({k:v for k,v in bad.items() if k!='sha256'})
        with self.pins(),self.assertRaisesRegex(ValueError,'partition'):
            m.build_inputs(self.original128,self.system256,bad,expected_predecessor_sha256=bad['sha256'])

    def test_resealed_tokens_gold_indices_and_calibration_cannot_change(self):
        for field in ('input_ids','raw_answer','index','documents','instruction','warmup'):
            bad=copy.deepcopy(self.artifacts);row=bad['manifest.json']['cohort'][-1]
            if field=='input_ids':row[field][0]+=1
            elif field=='documents':row[field][0]['input_end']-=1
            elif field=='warmup':bad['manifest.json']['warmup'][0]['input_ids'][0]+=1
            elif field=='index':row[field]=5
            else:row[field]='changed'
            self.reseal(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(bad)

    def test_resealed_offsets_source_documents_and_original_bundles_rejected(self):
        for field in ('offsets','source','original'):
            bad=copy.deepcopy(self.artifacts)
            if field=='offsets':bad['tokenization-evidence.json']['records'][-1]['token_offsets'].pop()
            elif field=='source':
                files=bad['source-records.json']['document_files'];files[next(iter(files))]+=' filler'
            else:
                old=bad['selection-plan.json']['original_input_bundles']['original128']['manifest.json']
                old['cohort'][-2]['raw_answer']='changed'
                old['sha256']=m.digest({k:v for k,v in old.items() if k!='sha256'})
            self.reseal(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(bad)

    def test_missing_extra_or_reordered_evaluations_rejected(self):
        for kind in ('missing','extra','reverse'):
            bad=copy.deepcopy(self.artifacts);rows=bad['manifest.json']['cohort']
            if kind=='missing':rows.pop()
            elif kind=='extra':rows.append(rows[0])
            else:rows.reverse()
            self.reseal(bad)
            with self.assertRaises(ValueError):self.check(bad)

    def test_existing_base_validator_receives_original_noncontiguous_indices(self):
        # Test callback delivery independently of deployment. The actual
        # external native validator is required again by runtime preflight.
        calls=[];manifests=[]
        from types import SimpleNamespace
        base=SimpleNamespace(validate_manifest=lambda v:manifests.append(v),document_specs=lambda r:calls.append(r['index']))
        self.assertTrue(self.check(base=base));self.assertEqual(calls,[0,1,2,3,4,20])
        self.assertEqual(manifests,[self.artifacts['manifest.json']])

    def test_cli_publishes_only_new_directory_and_validates_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp,self.pins():
            root=Path(temp);a=root/'old';b=root/'256';a.mkdir();b.mkdir()
            for folder,bundle in ((a,self.original128),(b,self.system256)):
                for n,value in bundle.items():m.publish(folder/n,value)
            receipt=root/'receipt.json';m.publish(receipt,self.predecessor)
            argv=['--previous128-input-dir',str(a),'--system256-input-dir',str(b),
                  '--predecessor-receipt',str(receipt),'--expected-predecessor-sha256',self.predecessor['sha256'],
                  '--output-dir',str(root/'out')]
            import io
            with contextlib.redirect_stdout(io.StringIO()):m.main(argv)
            self.assertTrue(m.validate_inputs(None,m.read(root/'out/manifest.json'),m.read(root/'out/input-protocol.json'),input_dir=root/'out'))
            with self.assertRaisesRegex(ValueError,'never overwrite'):m.main(argv)


if __name__=='__main__':unittest.main()
