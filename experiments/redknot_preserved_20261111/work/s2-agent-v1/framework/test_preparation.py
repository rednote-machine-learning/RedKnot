import importlib.util,json,pathlib,types,unittest,os
from unittest import mock
B=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('s2_adapter',B/'run_official.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class PreparationTests(unittest.TestCase):
 def test_real_root_option_uses_actual_public_field_only(self):
  calls=[]
  class constructor:
   def __init__(self,*args,**kwargs):calls.append((args,kwargs))
  module=types.SimpleNamespace(ApptainerWorkspace=constructor)
  with mock.patch.dict(os.environ,{},clear=True),a.admitted_environment():
   a.install_apptainer_root_option(module,0);module.ApptainerWorkspace(server_image='sha256:abc',working_dir='/workspace')
  kw=calls[0][1];self.assertFalse(kw['use_fakeroot']);self.assertFalse(kw['enable_gpu']);self.assertTrue(kw['enable_docker_compat']);self.assertIn('SESSION_API_KEY',kw['forward_env']);self.assertEqual(kw['disable_mount_locations'],['hostfs','bind-paths']);self.assertEqual(len(kw['extra_bind_mounts']),1);self.assertTrue(kw['extra_bind_mounts'][0].endswith(':/s2_runtime_guard:ro'))
 def test_unprivileged_root_option_rejected(self):
  with self.assertRaises(RuntimeError):a.install_apptainer_root_option(types.SimpleNamespace(),1000)
 def test_launch_plan_keeps_official_prompt_and_select(self):
  args=types.SimpleNamespace(llm_config='not-yet-created.json',output_dir='/tmp/s2-plan-test',arm='Full',workspace='apptainer',apptainer_real_root=True,run=False)
  plan=a.build_plan(args);self.assertIn('--enable-delegation',plan['argv']);self.assertNotIn('--disable-condenser',plan['argv']);self.assertTrue(plan['argv'][plan['argv'].index('--dataset')+1].endswith('agent_visible/tasks20.jsonl'));self.assertFalse(plan['custom_prompt_or_roles']);self.assertEqual(plan['status'],'plan_only')
 def test_safe_dataset_does_not_contain_grader_columns(self):
  s=json.loads((B.parent/'dataset/selection-manifest.json').read_text());rows=[json.loads(x) for x in (B.parent/'dataset/agent_visible/tasks20.jsonl').read_text().splitlines()];self.assertEqual(len(rows),20)
  for r in rows:self.assertEqual(set(r),set(s['agent_visible_fields']));self.assertFalse(set(r)&set(s['forbidden_agent_fields']))
if __name__=='__main__':unittest.main()
