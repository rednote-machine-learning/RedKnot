import copy,importlib.util,json,os,pathlib,tempfile,types,unittest
from unittest import mock
B=pathlib.Path(__file__).resolve().parent
import sys
sys.path.insert(0,str(B))
import run_official as a
import runtime_policy as p
import workspace_ownership as own
from routing_guard import validate_transport,validate_llm
class RuntimePolicyTests(unittest.TestCase):
 def args(self,ordinal):return types.SimpleNamespace(llm_config=str(B/'config/llm.Full.json'),output_dir='/tmp/s2-test',arm='Full',workspace='apptainer',apptainer_real_root=True,run=False,ordinal=ordinal)
 def test_subset_is_registered_ordinal_not_new_selection(self):
  ids=(B.parent/'dataset/selected_instances.txt').read_text().splitlines()
  for ordinal in [None,0,19]:
   plan=a.build_plan(self.args(ordinal));self.assertEqual(plan['registered_instances'],ids);self.assertEqual(plan['execution_instances'],ids if ordinal is None else [ids[ordinal]])
  for ordinal in [-1,20]:
   with self.assertRaises(ValueError):a.build_plan(self.args(ordinal))
 def test_equal_arm_config_and_fail_closed_overrides(self):
  for arm in ['Full','Prefix','S2']:self.assertEqual((B/'config/llm.Full.json').read_bytes(),(B/f'config/llm.{arm}.json').read_bytes())
  config=p.validate_config(B/'config/llm.Full.json');validate_llm(types.SimpleNamespace(**config))
  for k,v in [('base_url','https://example.org/v1'),('model','openai/gpt-5'),('max_output_tokens',32),('temperature',1),('fallback_strategy',object())]:
   updated={**config,k:v}
   with self.assertRaises(ValueError):validate_llm(types.SimpleNamespace(**updated))
 def test_transport_does_not_mutate_or_hide_caps(self):
  good={'model':'openai/deepseek-v4.1-flash','api_base':'http://127.0.0.1:8123/v1','temperature':0,'top_p':1,'seed':0,'tool_choice':'auto','messages':[{'role':'user','content':'test'}]}
  before=copy.deepcopy(good);self.assertIs(validate_transport(good),good);self.assertEqual(good,before)
  for k,v in [('max_tokens',32),('max_completion_tokens',32),('stop',['END']),('tool_choice','required'),('extra_body',{'model':'wrong'})]:
   with self.assertRaises(ValueError):validate_transport({**good,k:v})
 def test_environment_secrets_not_reported_and_cleanly_restored(self):
  with mock.patch.dict(os.environ,{'SESSION_API_KEY':'prior'},clear=True):
   with p.admitted_environment() as report:
    self.assertEqual(os.environ['APPTAINER_DROP_CAPS'],'all');self.assertNotIn(os.environ['SESSION_API_KEY'],json.dumps(report));self.assertEqual(len(report['session_api_key_sha256']),64)
   self.assertEqual(dict(os.environ),{'SESSION_API_KEY':'prior'})
 def test_environment_injection_rejected(self):
  for key in ['LLM_BASE_URL','OPENHANDS_APPTAINER_EXTRA_BINDS','APPTAINER_BIND','SINGULARITY_KEEP_PRIVS','APPTAINERENV_SECRET','OH_SESSION_API_KEYS_0']:
   with self.assertRaises(ValueError):p.validate_environment({key:'x'})
 def test_owned_process_record_and_cleanup_are_identity_bound(self):
  class Proc:
   pid=99
   def poll(self):return self.code
   code=None
  class Workspace:
   mount_dir='/tmp/task';_sif_path='/tmp/task.sif'
   def __init__(self,**kwargs):self._process=None;self._start_container()
   def _start_container(self):self._process=Proc()
   def cleanup(self):
    if self._process:self._process.code=0
    self._process=None
  identity={'pid':99,'start_ticks':123,'pgid':99,'sid':99,'argv_redacted':['apptainer','run','SESSION_API_KEY=<redacted>'],'cmdline_bytes_sha256':'x'}
  with tempfile.TemporaryDirectory() as td,mock.patch.dict(os.environ,{'SESSION_API_KEY':'secret-test'}),mock.patch.object(own,'proc_identity',return_value=identity):
   own.configure(td);C=own.make_workspace_class(Workspace,lambda kw:kw);w=C();w.cleanup();w.cleanup()
   docs=[json.loads(x.read_text()) for x in pathlib.Path(td,'workspace-processes').glob('*.json')]
   self.assertEqual({x['event'] for x in docs},{'started','ready','exited'});self.assertNotIn('secret-test',json.dumps(docs));self.assertEqual(next(x for x in docs if x['event']=='exited')['returncode'],0)
   w2=C.__new__(C);w2._process=Proc();w2._s2_owned_identity={**identity,'start_ticks':0}
   with self.assertRaises(RuntimeError):w2.cleanup()
if __name__=='__main__':unittest.main()
