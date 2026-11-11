import json,os,pathlib,tempfile,time,types,unittest
from unittest import mock
import run_triplet as m

class Process:
 def __init__(self,pid,code=None):self.pid=pid;self.returncode=code
 def poll(self):return self.returncode
class FakeBackend:
 def __init__(self,cfg,timeout_agent=False,fail_after_idle=False):self.cfg=cfg;self.next=90000;self.calls=[];self.services={};self.timeout_agent=timeout_agent;self.idles=0;self.fail_after_idle=fail_after_idle
 def spawn(self,argv,cwd,env,log_path):
  self.next+=1;p=Process(self.next);log=open(log_path,'x');ident={'pid':p.pid,'start_ticks':100+p.pid,'pgid':123,'sid':123,'boot_id':'fake','cmdline_bytes_sha256':'fake'};h=m.Handle(p,ident,log);h.argv=argv;self.calls.append(argv)
  if '--check-only' in argv:
   log.write(json.dumps({'status':'cpu_preflight_passed','gpu_initialized':False,'preflight':{'config_sha256':self.cfg['service_sha']}})+'\n');log.flush();p.returncode=0
  elif '-m' in argv:
   path=pathlib.Path(argv[argv.index('--output-dir')+1]);path.mkdir();h.directory=path;h.arm=argv[argv.index('--arm')+1];h.release=pathlib.Path(argv[argv.index('--release-file')+1]);h.deadline=float(argv[argv.index('--absolute-deadline-unix')+1]);self.services[p.pid]=h
   ready=pathlib.Path(argv[argv.index('--ready-file')+1]);m.publish(ready,{'phase':'pre_cuda_ready','status':'cpu-ready-no-torch-import','pid':p.pid,'manifest_sha256':self.cfg['service_sha'],'time_unix_s':time.time()})
  else:
   arm=argv[argv.index('--arm')+1];path=pathlib.Path(argv[argv.index('--output-dir')+1])/arm/'ordinal00'/'official';path.mkdir(parents=True)
   if not self.timeout_agent:
    (path/'output.jsonl').write_text(json.dumps({'instance_id':self.cfg['instance_id'],'error':None,'history':[{'kind':'ActionEvent','action':{'kind':'FinishAction','message':'CPU fake'}}]})+'\n');p.returncode=0
  return h
 def track(self,h):
  if hasattr(h,'release') and h.release.exists() and not (h.directory/'service-ready.json').exists():
   m.publish(h.directory/'service-ready.json',m.seal({'schema':'s2-online-service-ready-v1','ready':True,'pid':h.process.pid,'arm':h.arm,'config_sha256':self.cfg['service_sha'],'absolute_deadline_unix':h.deadline,'workers':{str(i):{'rank':i,'arm':h.arm,'config_sha256':self.cfg['service_sha']} for i in range(8)}}))
 def live(self,h):return h.process.poll() is None
 def signal(self,h):
  h.process.returncode=0
  if hasattr(h,'directory') and not (h.directory/'service-terminal.json').exists():m.publish(h.directory/'service-terminal.json',m.seal({'schema':'s2-online-service-terminal-v1','status':'stopped','arm':h.arm,'config_sha256':self.cfg['service_sha'],'absolute_deadline_unix':h.deadline,'errors':[],'survivors':[]}))
 def finish(self,h):self.signal(h);h.log.close();return []
 def workspace_cleanup(self,*args):return []
 def cleanup_cohort_remainders(self):return []
 def idle(self,cfg,deadline_monotonic=None):
  self.idles+=1
  if self.fail_after_idle and self.idles==2:raise RuntimeError('foreign context / cannot prove idle')
  return {'idle':True,'compute_apps':[],'gpus':[]}
 def health(self,arm):return True

class TripletTests(unittest.TestCase):
 def config(self,root):
  service=m.seal({'qualified_arms':['full','s2','prefix']});sp=root/'service-config.json';m.publish(sp,service);image=root/'image.json';m.publish(image,{'sif_path':'/fake/image.sif'})
  return {'sha256':'config-fixture','instance_id':'sphinx-doc__sphinx-8638','cwd':str(root),'service_python':'/fake/python','framework_python':'/fake/python','framework_script':'/fake/run_official.py','service_config':str(sp),'service_sha':service['sha256'],'image_provenance':str(image),'container_sif_sha256':'fake','llm_configs':{a:'/fake/common.json' for a,_ in m.ARMS},'environment':{'service':{},'framework':{}},'cpu_prepare_seconds':10,'cleanup_reserve_seconds':60,'service_startup_seconds':60}
 def execute(self,root,cfg,backend):
  ready=root/'ready.json';release=root/'release.json';original=m.publish
  def publish(path,value,**kwargs):
   original(path,value,**kwargs)
   if pathlib.Path(path)==ready:original(release,{'pid':os.getpid(),'released_unix':time.time()})
  with mock.patch.dict(os.environ,{'REDKNOT_GLM_RUN_ID':'cpu-fake'}),mock.patch.object(m,'validate_config',lambda *a,**k:cfg),mock.patch.object(m,'publish',publish):
   runner=m.Coordinator(cfg,root/'output',ready,release,backend);code=runner.run()
  return code,m.read_json(root/'output/triplet-terminal.json',sealed=True)
 def test_three_fresh_nested_releases_same_deadline_and_no_grade(self):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);cfg=self.config(root);b=FakeBackend(cfg);code,report=self.execute(root,cfg,b)
   self.assertEqual(code,0);self.assertEqual(report['completed_arms'],['Full','Prefix','S2']);self.assertFalse(report['CPU_grading_performed']);self.assertFalse(report['task_success_claim'])
   releases=[m.read_json(p,sealed=True) for p in (root/'output/arms').glob('*/nested-release.json')];self.assertEqual(len(releases),3);self.assertEqual(len({r['pid'] for r in releases}),3);self.assertEqual({r['absolute_deadline_unix'] for r in releases},{report['released_unix']+7200});self.assertEqual(b.idles,6)
   frameworks=[a for a in b.calls if '--ordinal' in a];self.assertEqual([a[a.index('--arm')+1] for a in frameworks],['Full','Prefix','S2']);self.assertTrue(all(a[a.index('--ordinal')+1]=='0' for a in frameworks))
 def test_wall_timeout_is_partial_and_still_cleans_before_next_arm(self):
  with tempfile.TemporaryDirectory() as td,mock.patch.object(m.Budget,'task_deadline',lambda self,reserve:time.time()-1):
   root=pathlib.Path(td);cfg=self.config(root);b=FakeBackend(cfg,timeout_agent=True);code,report=self.execute(root,cfg,b)
   self.assertEqual(code,1);self.assertEqual(report['completed_arms'],[]);self.assertEqual([r['status'] for r in report['arms']],['timed_out']*3);self.assertEqual(b.idles,6)
 def test_unverified_cleanup_prevents_following_arm(self):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);cfg=self.config(root);b=FakeBackend(cfg,fail_after_idle=True);code,report=self.execute(root,cfg,b)
   self.assertEqual(code,1);self.assertEqual(len(report['arms']),1);self.assertEqual(report['arms'][0]['status'],'cleanup_failed');self.assertEqual(report['not_started_arms'],['Prefix','S2'])
 def test_stale_nested_release_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);cfg=self.config(root);b=FakeBackend(cfg);c=m.Coordinator(cfg,root/'output',root/'r',root/'l',b);c.budget=m.Budget(time.time());release=root/'stale';release.write_text('{}')
   with self.assertRaises(ValueError):c.authorize_nested(types.SimpleNamespace(identity={'pid':123}),root/'no-ready',release,'sha')
   c.journal.close()
 def test_monotonic_guard_does_not_extend_7200_on_wall_clock_rollback(self):
  wall=[1000.];mono=[10.];b=m.Budget(1000,now=lambda:wall[0],monotonic=lambda:mono[0]);self.assertEqual(b.deadline,8200);wall[0]=900;mono[0]=7210;self.assertEqual(b.remaining(),0)
 def test_pid_reuse_and_exec_are_distinguished(self):
  a={'pid':12,'start_ticks':50,'pgid':12,'sid':12,'boot_id':'b','cmdline_bytes_sha256':'before'};b={**a,'cmdline_bytes_sha256':'after'}
  self.assertTrue(m.pc.same_process(a,b,argv=False));self.assertFalse(m.pc.same_process(a,b));self.assertFalse(m.pc.same_process(a,{**a,'start_ticks':51},argv=False))
 def test_workspace_reused_leader_rejected_before_enumeration(self):
  a={'pid':12,'start_ticks':50,'pgid':12,'sid':12,'boot_id':'b','cmdline_bytes_sha256':'a'}
  with mock.patch.object(m.pc,'identity',return_value={**a,'start_ticks':51}):
   with self.assertRaises(RuntimeError):m.pc.workspace_members(a)
 def test_orphan_sweep_requires_original_session_marker_and_birth_time(self):
  owner={'pid':10,'start_ticks':50,'pgid':10,'sid':10};child={'pid':20,'start_ticks':51,'pgid':10,'sid':10,'ppid':1}
  m.pc.authenticate_cohort_member(owner,child,True)
  for candidate,marker in [(child,False),({**child,'start_ticks':49},True),({**child,'sid':11},True),(owner,True)]:
   with self.assertRaises(RuntimeError):m.pc.authenticate_cohort_member(owner,candidate,marker)
if __name__=='__main__':unittest.main()
