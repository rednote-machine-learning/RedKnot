import json,pathlib,tempfile,types,unittest
import admitted_images as a
class AdmittedImageTests(unittest.TestCase):
 def fixture(self,root):
  image=root/'image.sif';image.write_bytes(b'CPU integrity fixture, not real container')
  row={'instance_id':'task-1','base_commit':'abc','image_tag_reference':'repo/task:latest'}
  value={'schema':'s2-official-image-preparation-v2','status':'image_built_or_existing','instance_id':'task-1','base_commit':'abc','base_image_tag':'docker.io/repo/task:latest','sif_path':str(image),'sif_sha256':a.file_sha(image),'sdk_commit':'pinned','target':'source-minimal','sphinx_dependency_wrap':False}
  proof=root/'proof.json';proof.write_text(json.dumps(value));return row,proof,value
 def test_exact_local_image_replaces_registry_or_build_without_fallback(self):
  with tempfile.TemporaryDirectory() as td:
   row,proof,value=self.fixture(pathlib.Path(td));images=a.load_images([proof],[row],'pinned')
   def never(*args,**kwargs):raise AssertionError('Registry/build must not run')
   module=types.SimpleNamespace(remote_image_exists=never,ensure_apptainer_agent_image=never);originals=a.install(module,images)
   self.assertFalse(module.remote_image_exists('any'))
   self.assertEqual(str(module.ensure_apptainer_agent_image(base_image=value['base_image_tag'],custom_tag='task',target='source-minimal',wrap_swebench_deps=False)),value['sif_path'])
   with self.assertRaises(ValueError):module.ensure_apptainer_agent_image(base_image='wrong',custom_tag='task',target='source-minimal',wrap_swebench_deps=False)
   a.restore(module,originals);self.assertIs(module.remote_image_exists,never)
 def test_modified_bytes_wrong_task_or_missing_proof_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   row,proof,value=self.fixture(pathlib.Path(td))
   with self.assertRaises(ValueError):a.load_images([], [row],'pinned')
   with self.assertRaises(ValueError):a.load_images([proof],[{**row,'base_commit':'other'}],'pinned')
   pathlib.Path(value['sif_path']).write_bytes(b'changed')
   with self.assertRaises(ValueError):a.load_images([proof],[row],'pinned')
if __name__=='__main__':unittest.main()
