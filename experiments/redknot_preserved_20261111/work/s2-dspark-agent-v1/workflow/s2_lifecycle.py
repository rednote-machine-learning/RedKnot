"""Thirty-minute native decode probe supervisor; holder stays OFF.

Reuses the pinned original cohort authentication and cleanup functions. This
supervisor owns the real monotonic timeout; no independent watchdog is claimed.
"""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,time,sys
LOCK='/tmp/redknot-nonprefix-scale-supervisor.lock'
PYTHON='/tmp/redknot-dsv41-sm90.6ZbGzo/di-venv/bin/python'
ORIGINAL='/tmp/redknot-dsv41-sm90.6ZbGzo/deepinfer_plan/glm53_gpu_lifecycle.py'
HELPER='/tmp/redknot-213-holder.4Ptbyc/run_segpaged_experiment.py'
PINS={ORIGINAL:'0a98c5a48d9f09e274be571df1c15b5988e09c8130ab516ed36436ff3690cafb',HELPER:'468375fbc05b7c69077663e4faef65ec11ac79297add994d087534723bd0accd'}
def require(x,msg):
 if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def seal(v):return dict(v,sha256=digest(v))
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(seal(v),f,indent=2);f.write('\n')
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def snapshot():
 def q(x):return subprocess.check_output(['nvidia-smi',x,'--format=csv,noheader,nounits'],text=True,timeout=20)
 rows=[]
 for line in q('--query-gpu=index,uuid,utilization.gpu,memory.used').splitlines():
  i,u,v,m=[x.strip()for x in line.split(',')];rows.append(dict(index=int(i),uuid=u,utilization=int(v),memory_mib=int(m)))
 apps=q('--query-compute-apps=pid,gpu_uuid,used_memory')
 require([x['index']for x in rows]==list(range(8)),'Exactly eight GPUs required')
 return dict(observed_unix=time.time(),gpus=rows,compute_apps=apps)
def idle(s):return not s['compute_apps'].strip() and all(x['memory_mib']==0 for x in s['gpus'])
def validate(c):
 require(c['sha256']==digest({k:v for k,v in c.items()if k!='sha256'}),'Config seal differs')
 require(c['schema']=='s2-dspark-agent-lifecycle-v1' and c['timeout_seconds']==7200 and c['holder_policy']=='OFF','User budget/holder policy differs')
 require(os.path.realpath(sys.executable)==os.path.realpath(PYTHON),'Fixed runtime required')
 for p,h in {**PINS,**c['source_sha256']}.items():require(sha(p)==h,'Source changed: '+p)
 require(c['source_sha256'].get(str(Path(__file__).resolve()))==sha(__file__),'Wrapper source unbound')
 require(c['source_sha256'].get(c['runner_argv'][0])==sha(c['runner_argv'][0]),'Probe unbound')
 require(not any(x.split('=',1)[0]in('--output-dir','--ready-file','--release-file')for x in c['runner_argv'][1:]),'Supervisor owns barrier/output paths')
 proof=read(c['cpu_admission']);require(proof.get('passed') is True and not proof.get('gpu_initialized',False),'CPU admission missing')
 require(c['source_sha256'].get(c['cpu_admission'])==sha(c['cpu_admission']),'CPU admission unbound')
 override=read(c['holder_off_receipt']);require(override['sha256']==digest({k:v for k,v in override.items()if k!='sha256'}) and override['status']=='holder_off_verified','Latest holder OFF receipt missing')
 require(not any(n=='torch' or n.startswith('deepinfer')for n in sys.modules),'Supervisor must not initialize inference')
def run(c):
 validate(c);original=load(ORIGINAL,'original_cleanup');helper=load(HELPER,'original_helper')
 out=Path(c['output_dir']);require(not out.exists(),'Fresh lifecycle output required')
 env=read(c['environment_json']);require('REDKNOT_GLM_RUN_ID' not in env,'Reserved owner marker')
 marker='s2-dspark-agent-'+c['sha256'][:20];env={**os.environ,**env,'REDKNOT_GLM_RUN_ID':marker,'SPT_NOENV':'1','PYTHONDONTWRITEBYTECODE':'1'}
 stop=[];previous={};child=None;log=None
 def request_stop(s,f):stop.append(s)
 with open(LOCK,'a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  before=snapshot();require(idle(before),'GPU must be physically idle; no holder or inference is stopped')
  out.mkdir();save(out/'before.json',before)
  state=dict(schema='s2-dspark-agent-lifecycle-terminal-v1',config_sha256=c['sha256'],holder_policy='OFF',status='failed',timeout_seconds=7200,independent_watchdog=False,started_unix=time.time())
  try:
   for s in (signal.SIGTERM,signal.SIGINT):previous[s]=signal.signal(s,request_stop)
   ready=out/'ready.json';release=out/'release.json'
   argv=[PYTHON,'-B','-u',*c['runner_argv'],'--output-dir',str(out/'benchmark'),'--ready-file',str(ready),'--release-file',str(release)]
   log=(out/'worker.log').open('x');child=subprocess.Popen(argv,cwd=c['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   stat=Path('/proc',str(child.pid),'stat').read_text().rsplit(') ',1)[1].split();state.update(child_pid=child.pid,child_start_ticks=int(stat[19]),child_argv=argv)
   save(out/'owner.json',state)
   until=time.monotonic()+600
   while not ready.exists():
    if stop:raise InterruptedError('User interruption during CPU preparation')
    if child.poll() is not None:raise RuntimeError('Probe exited before CPU-ready')
    if time.monotonic()>until:raise TimeoutError('CPU preparation timed out')
    time.sleep(.2)
   r=read(ready);require(r.get('pid')==child.pid and r.get('phase')=='pre_cuda_ready' and child.poll()is None,'CPU-ready identity differs')
   release_snapshot=snapshot();require(idle(release_snapshot),'GPU became occupied before release')
   state['released_unix']=time.time();state['deadline_unix']=state['released_unix']+7200
   save(release,dict(pid=child.pid,released_unix=state['released_unix'],timeout_seconds=7200,deadline_unix=state['deadline_unix'],holder_policy='OFF'))
   deadline=time.monotonic()+7200
   while child.poll() is None:
    if stop:raise InterruptedError('User interruption')
    if time.monotonic()>=deadline:raise TimeoutError('Authorized two-hour GPU budget exhausted')
    time.sleep(.5)
   state['runner_exit']=child.returncode
   require(child.returncode==0,'Probe returned nonzero')
   state['status']='complete'
  except BaseException as e:state.update(error=repr(e),status='timed_out' if isinstance(e,TimeoutError)else 'failed')
  finally:
   warnings=[]
   try:
    original.signal_owned_group(child,marker,signal.SIGTERM)
    if child:
     try:child.wait(timeout=20)
     except subprocess.TimeoutExpired:original.signal_owned_group(child,marker,signal.SIGKILL);child.wait(timeout=20)
     time.sleep(1);original.signal_owned_group(child,marker,signal.SIGKILL)
   except BaseException as e:warnings.append(repr(e))
   try:
    original.wait_for_quiescence(helper,child);after=snapshot();require(idle(after),'GPU contexts remain after cleanup');state['after']=after
    require([x['uuid']for x in after['gpus']]==[x['uuid']for x in before['gpus']],'GPU UUID changed')
    state['physical_idle_verified']=True
   except BaseException as e:state.update(cleanup_error=repr(e),physical_idle_verified=False,status='cleanup_failed')
   if log:log.close()
   for s,h in previous.items():signal.signal(s,h)
   state.update(cleanup_warnings=warnings,interruption_signals=stop,finished_unix=time.time(),holder_started=False)
   save(out/'terminal.json',state)
 print(json.dumps({k:state.get(k)for k in('status','physical_idle_verified','released_unix','deadline_unix','error')}))
 return 0 if state['status']=='complete'else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args();c=read(a.config)
 if a.check_only:validate(c);print(json.dumps(dict(status='lifecycle_cpu_check_passed',gpu_initialized=False,config_sha256=c['sha256'])))
 else:raise SystemExit(run(c))
