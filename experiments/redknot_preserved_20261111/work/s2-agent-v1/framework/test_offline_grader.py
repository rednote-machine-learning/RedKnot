"""CPU source/identity regressions; no container, network, GPU or gold execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import offline_grader as g


class OfflineGraderTests(unittest.TestCase):
    def row(self):
        return dict(instance_id=g.INSTANCE,base_commit=g.COMMIT,
            image_tag_reference='swebench/sweb.eval.x86_64.sphinx-doc_1776_sphinx-8638:latest')

    def base(self, root):
        image=(root/'base.sif').resolve();image.write_bytes(b'CPU fixture only, not a SIF')
        manifest=g.B.parent/'container/oci/swebench_sweb.eval.x86_64.sphinx-doc_1776_sphinx-8638/manifest.raw.json'
        value=g.seal(dict(schema='s2-official-base-image-cpu-proof-v1',status='passed',
            instance_id=g.INSTANCE,base_commit=g.COMMIT,sif_path=str(image),sif_sha256=g.file_sha(image),
            archive_sha256=g.ARCHIVE_SHA,registry_manifest_digest=g.MANIFEST_DIGEST,
            steps=[dict(argv=['apptainer','build',str(image),'docker-archive:///verified/archive.tar'],returncode=0),
                   dict(argv=['apptainer','exec',str(image),'git','-C','/testbed','rev-parse','HEAD'],returncode=0,stdout=g.COMMIT+'\n')]))
        receipt=root/'proof.json';receipt.write_text(json.dumps(value))
        return receipt,manifest,value

    def test_exact_official_source_sif_and_commit_required(self):
        with tempfile.TemporaryDirectory() as name:
            proof,manifest,value=self.base(Path(name))
            admitted=g.admitted_base(proof,manifest,self.row())
            self.assertEqual(admitted['archive_sha256'],g.ARCHIVE_SHA)
            for field,new in [('status','failed'),('base_commit','different'),('archive_sha256','0'*64)]:
                changed={k:v for k,v in value.items() if k!='sha256'};changed[field]=new
                proof.write_text(json.dumps(g.seal(changed)))
                with self.assertRaises(ValueError):g.admitted_base(proof,manifest,self.row())
            proof.write_text(json.dumps(value));Path(value['sif_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes changed'):g.admitted_base(proof,manifest,self.row())

    def test_only_transport_hook_is_changed_and_restored_on_failure(self):
        row=self.row();original=lambda instance:'docker://'+row['image_tag_reference']
        module=NS(image_uri=original,score_instance=object(),score_shell=object(),verify_sandbox=object())
        before=dict(vars(module))
        with self.assertRaisesRegex(RuntimeError,'test failure'):
            with g.local_image_uri(module,row,dict(image_path='/verified/base.sif')):
                self.assertEqual(module.image_uri(row),'/verified/base.sif')
                self.assertIs(module.score_instance,before['score_instance'])
                with self.assertRaises(ValueError):module.image_uri(dict(row,base_commit='changed'))
                with self.assertRaises(ValueError):module.image_uri(dict(row,instance_id='other'))
                raise RuntimeError('test failure')
        self.assertEqual(vars(module),before)

    def test_explicit_official_setup_derivation_preserves_original_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);proof,manifest,value=self.base(root)
            original={k:v for k,v in value.items() if k!='sha256'}
            original['status']='failed';original['steps'][1]['stdout']=g.ENVIRONMENT_COMMIT+'\n'
            original=g.seal(original);old=root/'original-failed.json';old.write_text(json.dumps(original))
            evidence=root/'official-setup.json';evidence.write_text(json.dumps(dict(cpu_fixture=True)))
            derived=g.seal(dict(schema='s2-official-base-environment-admission-v1',status='passed',
                instance_id=g.INSTANCE,dataset_base_commit=g.COMMIT,baseline_parent_commit=g.COMMIT,
                baseline_environment_commit=g.ENVIRONMENT_COMMIT,baseline_environment_tree=g.ENVIRONMENT_TREE,
                approved_changed_paths=['setup.py','tox.ini'],setup_delta_sha256=g.SETUP_DELTA_SHA,
                official_harness_evidence=dict(path=str(evidence),file_sha256=g.file_sha(evidence)),
                original_base_proof_ref=dict(path=str(old),file_sha256=g.file_sha(old),sha256=original['sha256']),
                sif_path=value['sif_path'],sif_sha256=value['sif_sha256'],archive_sha256='sha256:'+g.ARCHIVE_SHA,
                registry_manifest_digest=g.MANIFEST_DIGEST))
            proof.write_text(json.dumps(derived));result=g.admitted_base(proof,manifest,self.row())
            self.assertEqual(result['expected_environment_tree'],g.ENVIRONMENT_TREE)
            self.assertEqual(json.loads(old.read_text())['status'],'failed')
            evidence.write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence bytes changed'):
                g.admitted_base(proof,manifest,self.row())

    def test_original_image_resolution_cannot_be_bypassed(self):
        module=NS(image_uri=lambda instance:'docker://unrelated:latest')
        with g.local_image_uri(module,self.row(),dict(image_path='/verified/base.sif')):
            with self.assertRaisesRegex(ValueError,'TestSpec'):module.image_uri(self.row())
        with patch.dict('os.environ',{'OPENHANDS_SWEBENCH_IMAGE_TEMPLATE':'override'}):
            with self.assertRaisesRegex(ValueError,'template'):
                with g.local_image_uri(module,self.row(),dict(image_path='/verified/base.sif')):pass

    def test_actual_official_sandbox_builder_uses_only_local_source(self):
        sdk=ModuleType('openhands.sdk');sdk.get_logger=lambda name:NS(info=lambda *a:None)
        constants=ModuleType('swebench.harness.constants');constants.KEY_INSTANCE_ID='instance_id'
        path=g.B/'benchmarks/benchmarks/swebench/apptainer_eval.py'
        spec=importlib.util.spec_from_file_location('actual_official_apptainer_fixture',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'openhands.sdk':sdk,'swebench.harness.constants':constants}):
            spec.loader.exec_module(module)
            original_score=module.score_instance
            module.image_uri=lambda instance:'docker://'+self.row()['image_tag_reference']
            module.shutil=NS(which=lambda command:'/fixture/apptainer')
            seen=[]
            def run(command,*args,**kw):
                seen.append(command);Path(command[-2]).mkdir();return NS(returncode=0,stdout='')
            module._run=run
            with tempfile.TemporaryDirectory() as name:
                root=Path(name);(root/'scores').mkdir();(root/'sandboxes').mkdir()
                with g.local_image_uri(module,self.row(),dict(image_path='/verified/base.sif')):
                    sandbox=module.ensure_sandbox(self.row(),root/'scores',root/'sandboxes',root/'cache')
                self.assertEqual(seen[0][-1],'/verified/base.sif')
                self.assertTrue(sandbox.is_dir());self.assertIs(module.score_instance,original_score)

    def test_pregrade_probe_asserts_exact_head_without_reset_or_patch(self):
        commands=[]
        module=NS(apptainer_base_cmd=lambda sandbox,work:['apptainer','exec',str(sandbox),'bash','-lc'],
            _run=lambda command,*a,**k:(commands.append(command) or NS(returncode=0,stdout=g.COMMIT)))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            report=g.verify_fresh_sandbox(module,self.row(),root/'sandbox',root,root/'cache')
            self.assertTrue(report['tracked_tree_pristine'])
            script=commands[0][-1]
            self.assertIn('test "$(git rev-parse HEAD)" = '+g.COMMIT,script)
            self.assertIn('conda activate testbed',script)
            self.assertNotIn('git reset',script);self.assertNotIn('git apply',script)

    def test_network_isolation_rejects_unavailable_platform_before_grading(self):
        with patch.object(sys,'platform','darwin'):
            with self.assertRaisesRegex(ValueError,'Linux real-root'):g.isolate_network()

    def test_environment_baseline_is_verified_without_destroying_setup_commit(self):
        seen=[]
        module=NS(apptainer_base_cmd=lambda sandbox,work:['apptainer','exec',str(sandbox),'bash','-lc'],
            _run=lambda command,*a,**k:(seen.append(command[-1]) or NS(returncode=0,stdout='verified')))
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            g.verify_fresh_sandbox(module,self.row(),root/'sandbox',root,root/'cache',base=dict(
                expected_environment_commit=g.ENVIRONMENT_COMMIT,expected_environment_tree=g.ENVIRONMENT_TREE))
            self.assertIn(g.ENVIRONMENT_COMMIT,seen[0]);self.assertIn(g.ENVIRONMENT_TREE,seen[0])
            self.assertNotIn('git reset',seen[0]);self.assertNotIn('git checkout',seen[0])


if __name__=='__main__':unittest.main()
