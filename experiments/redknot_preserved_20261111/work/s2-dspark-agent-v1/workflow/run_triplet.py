"""Independent DSpark-enabled Full -> Prefix -> S2 for registered task0.

Stdlib-only before CPU-ready/release. Never starts/stops a holder or grades patches.
Every native service has a fresh nested barrier and shares the new authorized round
release+7200 deadline. Exact cleanup and empty pinned GPUs
precede arm switch. No framework, service, model or task implementation changes.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,signal,subprocess,sys,tempfile,time,urllib.request
from dataclasses import dataclass,field
import process_control as pc
ARMS=(('Full','full'),('Prefix','prefix'),('S2','s2'))
RELEASE_MARGIN_SECONDS=180
CLEANUP_RESERVE_SECONDS=60
MINIMUM_GPU_SECONDS=300

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def file_sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read_json(path,sealed=False):
    value=json.loads(pathlib.Path(path).read_text())
    if sealed and value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):raise ValueError('Evidence seal differs: '+str(path))
    return value

def publish(path,value,*,exclusive=True):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.s2-publish-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        if exclusive:os.link(tmp,path)
        else:os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def required(condition,message):
    if not condition:raise ValueError(message)

def nested(value,key):
    for part in key.split('.'):value=value[part]
    return value

def validate_config(config,*,check_files=True):
    required(config.get('schema')=='s2-dspark-agent-triplet-config-v1','Wrong triplet retry schema')
    required(config.get('sha256')==digest({k:v for k,v in config.items() if k!='sha256'}),'Triplet config seal differs')
    required(config.get('arm_order')==['Full','Prefix','S2'] and config.get('ordinal')==0,'Fixed first registered task and three-arm order required')
    required(config.get('maximum_round_seconds')==7200 and config.get('task_wall_seconds')==1800,'Only original round7200 and uniform1800 task budget admitted')
    required(config.get('holder_policy')=='OFF' and config.get('decode_profile')=='dspark','Enabled decode and holder OFF required')
    required('timeout_seconds' not in config,'Supervisor owns the GPU timeout')
    required(config.get('cleanup_reserve_seconds')==CLEANUP_RESERVE_SECONDS,'Original absolute deadline requires60s cleanup reserve')
    required(1<=config.get('service_startup_seconds',0)<=1800,'Bounded service startup required')
    required(1<=config.get('cpu_prepare_seconds',0)<=600,'Bounded CPU preparation required')
    required(set(config.get('gpu_uuid_by_index',{}))==set(map(str,range(8))) and len(set(config['gpu_uuid_by_index'].values()))==8,'Exactly eight pinned GPU UUIDs required')
    required(set(config.get('llm_configs',{}))=={'Full','S2','Prefix'},'All three LLM configurations required')
    required(set(config.get('environment',{}))=={'service','framework'},'Explicit per-role environment required')
    for env in config['environment'].values():
        required(isinstance(env,dict) and all(isinstance(k,str) and k and '=' not in k and isinstance(v,str) and '\0' not in k+v for k,v in env.items()),'Invalid environment')
        required('REDKNOT_GLM_RUN_ID' not in env,'Original lifecycle marker cannot be overwritten')
    for key in ['cwd','service_python','framework_python','service_config','framework_script','image_provenance','selection_manifest']:
        required(pathlib.Path(config[key]).is_absolute(),'Absolute path required: '+key)
    sources=config.get('source_sha256',{})
    essentials={str(pathlib.Path(__file__).resolve()),str(pathlib.Path(pc.__file__).resolve()),config['framework_script'],config['service_config'],config['image_provenance'],config['selection_manifest'],*config['llm_configs'].values()}
    framework=pathlib.Path(config.get('framework_pinned_source_dir',str(pathlib.Path(config['framework_script']).parent)))
    if 'framework_pinned_source_dir' in config:
        required(framework.is_absolute(),'Pinned framework source directory must be absolute')
        essentials.add(str(pathlib.Path(config['framework_script']).parent/'runtime_policy.py'))
    essentials.update(str(framework/name) for name in ('runtime_policy.py','admitted_images.py','workspace_ownership.py','runtime_guard/routing_guard.py','runtime_guard/sitecustomize.py'))
    required(essentials<=set(sources),'Missing essential source/config/image pins')
    refs=config.get('admission_refs',[])
    required({'source','container','sdk_wire'}<={r.get('kind') for r in refs},'Source, actual container and SDK-wire CPU admissions required')
    for ref in refs:
        required(bool(ref.get('required_fields')) and isinstance(ref['required_fields'],dict),'Explicit CPU proof fields required')
    if check_files:
        for path,sha in sources.items():required(file_sha(path)==sha,'Pinned file changed: '+path)
        for ref in refs:
            required(file_sha(ref['path'])==ref['file_sha256'],'CPU admission bytes changed')
            value=read_json(ref['path'],sealed=bool(ref.get('seal_sha256')))
            if ref.get('seal_sha256'):required(value['sha256']==ref['seal_sha256'],'CPU admission seal changed')
            required(all(nested(value,k)==v for k,v in ref['required_fields'].items()),'CPU admission required fields not satisfied')
        configs=[pathlib.Path(config['llm_configs'][name]).read_bytes() for name,_ in ARMS]
        required(configs[0]==configs[1]==configs[2],'Three arms must use identical SDK model/sampling configs')
        service=read_json(config['service_config'],sealed=True)
        required(set(service['qualified_arms'])=={'full','s2','prefix'},'All three service arms must be registered before release')
        required(service['host']=='127.0.0.1' and service['port']==8123,'Fixed local endpoint required')
        selection=read_json(config['selection_manifest'],sealed=True)
        required(selection['selected_instance_ids'][0]==config['instance_id'],'Task0 differs from sealed registered selection')
        image=read_json(config['image_provenance'])
        required(image.get('instance_id')==config['instance_id'] and image.get('sif_sha256')==config['container_sif_sha256'],'Image differs from registered task/admitted SIF')
        required(file_sha(image['sif_path'])==image['sif_sha256'],'Actual SIF changed since CPU admission')
    return config

class Budget:
    def __init__(self,released,now=time.time,monotonic=time.monotonic):
        required(type(released) in (int,float) and math.isfinite(released),'Invalid release timestamp')
        self.released= released;self.deadline=released+7200;self.now=now;self.monotonic=monotonic
        self.monotonic_deadline=monotonic()+max(0,self.deadline-now())
    def remaining(self):return max(0,min(self.deadline-self.now(),self.monotonic_deadline-self.monotonic()))
    def task_deadline(self,reserve):return min(self.now()+1800,self.deadline-reserve)

@dataclass
class Handle:
    process:object
    identity:dict
    log:object
    descendants:dict=field(default_factory=dict)

class Backend:
    def __init__(self,marker):self.marker=marker;self.coordinator=pc.identity(os.getpid())
    def spawn(self,argv,cwd,env,log_path):
        log=open(log_path,'x');p=subprocess.Popen(argv,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=False)
        try:
            until=time.monotonic()+5
            while True:
                if p.poll() is not None:raise RuntimeError('Child exited before identity capture: '+str(p.returncode))
                info=pc.identity(p.pid)
                if info['cmdline_bytes_sha256']==hashlib.sha256(b'\0'.join(x.encode() for x in argv)+b'\0').hexdigest():break
                if time.monotonic()>=until:raise RuntimeError('Child argv did not match requested command')
                time.sleep(.02)
            required(info['pgid']==os.getpgrp() and info['sid']==os.getsid(0),'Child detached from original lifecycle session')
            required(pc.has_marker(p.pid,self.marker),'Child lacks original lifecycle ownership marker')
            return Handle(p,info,log)
        except BaseException:
            p.terminate();p.wait(timeout=5);log.close();raise
    def track(self,handle):
        if handle is None or handle.process.poll() is not None:return
        for value in pc.descendants(handle.identity,self.marker):
            prior=handle.descendants.get(value['pid'])
            if prior and not pc.same_process(prior,value,argv=False):raise RuntimeError('Tracked child PID changed')
            handle.descendants[value['pid']]=value
    def live(self,handle):return handle.process.poll() is None
    def signal(self,handle):
        self.track(handle)
        if self.live(handle):pc.signal_exact(handle.identity,signal.SIGTERM,argv=False)
    def finish(self,handle):
        if handle is None:return []
        self.track(handle)
        actions=pc.stop_exact_set([*handle.descendants.values(),handle.identity],grace=3)
        handle.process.wait(timeout=3);handle.log.close();return actions
    def idle(self,config,deadline_monotonic=None):return pc.gpu_snapshot(config['gpu_uuid_by_index'],deadline_monotonic=deadline_monotonic)
    def cleanup_cohort_remainders(self):
        reports=[]
        for attempt in range(3):
            members=pc.cohort_members(self.coordinator,self.marker)
            if not members:break
            reports.append({'authentication':'original_lifecycle_pgid_sid_private_marker_and_startticks','members':members,'actions':pc.stop_exact_set(members,grace=3)})
        required(not pc.cohort_members(self.coordinator,self.marker),'Owned cohort still has live processes')
        return reports
    def health(self,arm):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8123/health',timeout=2) as f:value=json.load(f)
            return value.get('ready') is True and value.get('arm')==arm and value.get('concurrency')==1 and value.get('failure') is None
        except (OSError,ValueError):return False
    def workspace_cleanup(self,root,owner_pid,sif_path):
        roots={};exits=set();reports=[]
        for path in sorted(pathlib.Path(root).rglob('workspace-processes/*.json')):
            value=read_json(path);required(value.get('schema')=='s2-workspace-process-ownership-v1' and value.get('owner_pid')==owner_pid,'Workspace ownership receipt detached')
            required(str(pathlib.Path(value['sif_path']).resolve())==str(pathlib.Path(sif_path).resolve()),'Workspace used a different SIF')
            info=value['process'];key=(info['pid'],info['start_ticks'])
            required(info['pgid']==info['pid'] and info['sid']==info['pid'],'Workspace must be its own recorded session')
            if value['event']=='exited':exits.add(key)
            elif value['event'] in ('started','ready'):
                previous=roots.get(key)
                if previous is None or value['event']=='ready':roots[key]=value
        for key,value in roots.items():
            info=value['process']
            try:current=pc.identity(info['pid'])
            except (FileNotFoundError,ProcessLookupError):current=None
            if current is not None:
                required(pc.same_process(info,current,argv=value['event']=='ready'),'Workspace PID/ready identity changed')
            actions=[]
            for attempt in range(3):
                members=pc.workspace_members(info)
                if not members:break
                actions+=pc.stop_exact_set(members,grace=3)
            required(not pc.workspace_members(info),'Workspace session still has live members')
            reports.append({'leader':info,'latest_event':value['event'],'already_recorded_exit':key in exits,'actions':actions})
        return reports

class Coordinator:
    def __init__(self,config,output,ready,release,backend=None):
        self.config=config;self.output=pathlib.Path(output);self.ready=pathlib.Path(ready);self.release=pathlib.Path(release)
        self.marker=os.environ.get('REDKNOT_GLM_RUN_ID','');self.backend=backend or Backend(self.marker)
        self.stopping=False;self.budget=None;self.events=[];self.arms=[];self.active=[]
        self.output.mkdir(parents=True,exist_ok=False)
        self.journal=(self.output/'events.jsonl').open('x',buffering=1)
    def event(self,name,**data):
        row=seal({'schema':'s2-triplet-event-v1','ordinal':len(self.events),'event':name,'observed_unix':time.time(),'previous_sha256':self.events[-1]['sha256'] if self.events else None,**data})
        self.events.append(row);self.journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');self.journal.flush()
        publish(self.output/'progress.json',seal({'schema':'s2-triplet-progress-v1','latest_event':row,'arms':self.arms,'absolute_deadline_unix':self.budget.deadline if self.budget else None}),exclusive=False)
    def check(self):
        if self.stopping:raise InterruptedError('Original lifecycle/user termination requested')
        if self.budget and self.budget.remaining()<=self.config['cleanup_reserve_seconds']:raise TimeoutError('Shared deadline reached cleanup reserve')
    def wait(self,predicate,deadline,handle=None):
        while True:
            self.check()
            if handle:
                self.backend.track(handle)
                if not self.backend.live(handle):raise RuntimeError('Child exited before required phase: '+str(handle.process.returncode))
            if predicate():return
            if time.time()>=deadline:raise TimeoutError('Phase deadline expired')
            time.sleep(.2)
    def environment(self,role):
        env=os.environ.copy();env.update(self.config['environment'][role]);env['REDKNOT_GLM_RUN_ID']=self.marker;return env
    def service_command(self,arm):return [self.config['service_python'],'-u','-m','service.serve','--config',self.config['service_config'],'--arm',arm]
    def prepare(self):
        required(bool(self.marker),'Original supervisor ownership marker required')
        required(not self.ready.exists() and not self.release.exists(),'Outer barrier sentinel already exists')
        validate_config(self.config)
        publish(self.output/'triplet-config.json',self.config)
        until=time.time()+self.config['cpu_prepare_seconds']
        for label,arm in ARMS:
            log=self.output/f'cpu-preflight-{arm}.log'
            proc=self.backend.spawn(self.service_command(arm)+['--check-only'],self.config['cwd'],self.environment('service'),log)
            self.active.append(proc)
            try:
                while self.backend.live(proc):
                    self.check()
                    if time.time()>=until:raise TimeoutError('CPU admission command timeout')
                    time.sleep(.1)
                required(proc.process.returncode==0,'Actual service CPU preflight failed: '+arm)
            finally:self.backend.finish(proc)
            reports=[]
            for line in log.read_text().splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict) and value.get('status')=='cpu_preflight_passed':reports.append(value)
            required(len(reports)==1 and reports[0].get('gpu_initialized') is False and reports[0].get('preflight',{}).get('config_sha256')==read_json(self.config['service_config'])['sha256'],'Actual CPU preflight report missing/detached')
            self.event('cpu_service_preflight_passed',arm=label,log_sha256=file_sha(log))
        self.event('cpu_cohort_quiescent',cleanup=self.backend.cleanup_cohort_remainders())
        ready_time=time.time()
        publish(self.ready,seal({'schema':'s2-triplet-cpu-ready-v1','pid':os.getpid(),'phase':'pre_cuda_ready','status':'cpu-ready-no-torch-import','config_sha256':self.config['sha256'],'time_unix_s':ready_time}))
        self.event('awaiting_original_release',ready_file=str(self.ready))
        self.wait(lambda:self.release.exists(),time.time()+self.config['cpu_prepare_seconds'])
        value=read_json(self.release)
        required(value.get('pid')==os.getpid() and type(value.get('released_unix')) in (int,float) and math.isfinite(value['released_unix']) and ready_time<=value['released_unix']<=time.time()+2,'Outer release does not authorize this CPU-ready process')
        self.budget=Budget(value['released_unix'])
        validate_config(self.config)
        self.event('original_release_accepted',release_file_sha256=file_sha(self.release),released_unix=self.budget.released,absolute_deadline_unix=self.budget.deadline)
    def idle_wait(self,label):
        end=time.monotonic()+20;last=None;timeouts=[]
        while time.monotonic()<end:
            try:last=self.backend.idle(self.config,deadline_monotonic=end)
            except subprocess.TimeoutExpired as exc:
                timeouts.append({'command':exc.cmd,'timeout_seconds':exc.timeout})
                self.event('gpu_idle_query_timeout',phase=label,attempt=len(timeouts),timeout_seconds=exc.timeout)
                continue
            if last['idle']:
                required(time.monotonic()<=end,'GPU idle evidence arrived after20s verification window')
                self.event('gpu_idle_verified',phase=label,snapshot=last,query_timeouts=timeouts);return last
            time.sleep(min(.5,max(0,end-time.monotonic())))
        self.event('gpu_idle_not_verified',phase=label,snapshot=last,query_timeouts=timeouts)
        raise RuntimeError('GPU contexts or allocated memory remain; refusing next arm')
    def authorize_nested(self,handle,ready,release,config_sha):
        required(not pathlib.Path(release).exists(),'Nested release already exists')
        value=read_json(ready)
        required(value.get('pid')==handle.identity['pid'] and value.get('phase')=='pre_cuda_ready' and value.get('status')=='cpu-ready-no-torch-import' and value.get('manifest_sha256')==config_sha,'Nested CPU ready is detached')
        required(type(value.get('time_unix_s')) in (int,float) and value['time_unix_s']<=time.time(),'Nested ready timestamp invalid')
        self.check();self.backend.track(handle);required(self.backend.live(handle),'Service exited before nested release')
        release_value=seal({'schema':'s2-nested-service-release-v1','pid':handle.identity['pid'],'released_unix':time.time(),'round_released_unix':self.budget.released,'retry_released_unix':self.budget.released,'absolute_deadline_unix':self.budget.deadline,'coordinator_pid':os.getpid(),'coordinator_config_sha256':self.config['sha256'],'service_config_sha256':config_sha,'ready_file_sha256':file_sha(ready)})
        publish(release,release_value);self.event('nested_service_released',service_identity=handle.identity,release=release_value)
    def read_outcome(self,arm_root):
        paths=list(pathlib.Path(arm_root).rglob('output.jsonl'))
        required(len(paths)==1,'Exactly one official output.jsonl expected')
        rows=[json.loads(x) for x in paths[0].read_text().splitlines() if x.strip()]
        required(rows and all(r.get('instance_id')==self.config['instance_id'] for r in rows),'Official output task identity differs')
        last=rows[-1];finished=any(isinstance(e,dict) and isinstance(e.get('action'),dict) and e['action'].get('kind')=='FinishAction' for e in last.get('history',[]))
        return {'path':str(paths[0]),'file_sha256':file_sha(paths[0]),'records':len(rows),'error':last.get('error'),'finish_action_observed':finished,'execution_complete':last.get('error') is None and finished,'task_success':None,'grading':'official CPU grading after GPU cleanup; holder stays OFF'}
    def run_arm(self,label,arm):
        self.check();validate_config(self.config);self.idle_wait('before-'+arm)
        directory=self.output/'arms'/label;directory.mkdir(parents=True,exist_ok=False)
        ready=directory/'nested-ready.json';release=directory/'nested-release.json';service_out=directory/'service'
        required(not any(p.exists() for p in [ready,release,service_out]),'Nested service paths must be fresh')
        service_cfg=read_json(self.config['service_config'],sealed=True);config_sha=service_cfg['sha256']
        record={'arm':label,'service_arm':arm,'instance_id':self.config['instance_id'],'started_unix':time.time(),'status':'starting','absolute_deadline_unix':self.budget.deadline,'task_success':None}
        self.arms.append(record);service=None;agent=None;cleanup_errors=[]
        agent_root=self.output/'agent_runs';expected_agent_root=agent_root/label/'ordinal00'
        try:
            argv=self.service_command(arm)+['--run','--output-dir',str(service_out),'--ready-file',str(ready),'--release-file',str(release),'--absolute-deadline-unix',str(self.budget.deadline)]
            service=self.backend.spawn(argv,self.config['cwd'],self.environment('service'),directory/'service.log');self.active.append(service)
            record['service_identity']=service.identity;self.event('service_started',arm=label,identity=service.identity)
            start_deadline=min(time.time()+self.config['service_startup_seconds'],self.budget.deadline-self.config['cleanup_reserve_seconds'])
            self.wait(lambda:ready.exists(),start_deadline,service);self.authorize_nested(service,ready,release,config_sha)
            endpoint_ready=service_out/'service-ready.json';self.wait(lambda:endpoint_ready.exists(),start_deadline,service)
            value=read_json(endpoint_ready,sealed=True)
            required(value.get('schema')=='s2-online-service-ready-v1' and value.get('ready') is True and value.get('pid')==service.identity['pid'] and value.get('config_sha256')==config_sha and value.get('arm')==arm and value.get('absolute_deadline_unix')==self.budget.deadline,'Endpoint-ready identity/policy differs')
            required(set(value.get('workers',{}))==set(map(str,range(8))),'Endpoint requires all8 ready ranks')
            required(all(m.get('rank')==int(rank) and m.get('arm')==arm and m.get('config_sha256')==config_sha for rank,m in value['workers'].items()),'Endpoint worker identities differ')
            self.wait(lambda:self.backend.health(arm),start_deadline,service)
            record['service_ready_unix']=time.time();record['service_ready_file_sha256']=file_sha(endpoint_ready)
            self.event('service_endpoint_ready',arm=label,ready_file_sha256=file_sha(endpoint_ready))
            argv=[self.config['framework_python'],'-u',self.config['framework_script'],'--llm-config',self.config['llm_configs'][label],'--output-dir',str(agent_root),'--arm',label,'--workspace','apptainer','--apptainer-real-root','--ordinal','0','--image-provenance',self.config['image_provenance'],'--run']
            agent=self.backend.spawn(argv,self.config['cwd'],self.environment('framework'),directory/'framework.log');self.active.append(agent)
            task_deadline=self.budget.task_deadline(self.config['cleanup_reserve_seconds']);record.update(agent_identity=agent.identity,agent_started_unix=time.time(),task_deadline_unix=task_deadline,task_wall_limit_seconds=1800,status='agent_running')
            self.event('agent_started',arm=label,identity=agent.identity,task_deadline_unix=task_deadline)
            while self.backend.live(agent):
                self.check();self.backend.track(agent);self.backend.track(service)
                if not self.backend.live(service):raise RuntimeError('Native service exited during agent task')
                if time.time()>=task_deadline:raise TimeoutError('Task wall budget exhausted')
                time.sleep(.2)
            record['agent_finished_unix']=time.time();record['framework_exit_code']=agent.process.returncode
            required(agent.process.returncode==0,'Official framework exited nonzero')
            record['outcome']=self.read_outcome(expected_agent_root)
            record['status']='complete_awaiting_cpu_grading' if record['outcome']['execution_complete'] else 'incomplete_agent_output'
        except BaseException as exc:
            record['status']='timed_out' if isinstance(exc,TimeoutError) else 'interrupted' if isinstance(exc,InterruptedError) else 'failed'
            record['error']=repr(exc);self.event('arm_error',arm=label,error=repr(exc))
        finally:
            # First ask the framework to stop; its handler closes official workspaces.
            if agent:
                try:
                    self.backend.signal(agent)
                    end=time.monotonic()+8
                    while self.backend.live(agent) and time.monotonic()<end:time.sleep(.1)
                    record['workspace_cleanup']=self.backend.workspace_cleanup(expected_agent_root,agent.identity['pid'],read_json(self.config['image_provenance'])['sif_path'])
                    record['agent_cleanup']=self.backend.finish(agent)
                except BaseException as exc:cleanup_errors.append('framework: '+repr(exc))
            if service:
                try:
                    self.backend.signal(service)
                    end=time.monotonic()+12
                    while self.backend.live(service) and time.monotonic()<end:time.sleep(.1)
                    record['service_cleanup']=self.backend.finish(service)
                    terminal=service_out/'service-terminal.json'
                    if terminal.exists():
                        terminal_value=read_json(terminal,sealed=True);record['service_terminal']={'file_sha256':file_sha(terminal),'status':terminal_value.get('status'),'errors':terminal_value.get('errors'),'survivors':terminal_value.get('survivors')}
                        required(terminal_value.get('schema')=='s2-online-service-terminal-v1' and terminal_value.get('arm')==arm and terminal_value.get('config_sha256')==config_sha and terminal_value.get('absolute_deadline_unix')==self.budget.deadline,'Service terminal identity differs')
                        if terminal_value.get('errors') or terminal_value.get('survivors'):record['status']='service_failed'
                    else:record['status']='service_terminal_missing'
                except BaseException as exc:cleanup_errors.append('service: '+repr(exc))
            try:record['cohort_remainder_cleanup']=self.backend.cleanup_cohort_remainders()
            except BaseException as exc:cleanup_errors.append('original cohort: '+repr(exc))
            try:record['post_arm_idle']=self.idle_wait('after-'+arm)
            except BaseException as exc:cleanup_errors.append('GPU idle: '+repr(exc))
            record['finished_unix']=time.time();record['cleanup_errors']=cleanup_errors
            if cleanup_errors:record['status']='cleanup_failed'
            publish(directory/'arm-terminal.json',seal(record));self.event('arm_terminal',arm=label,status=record['status'],cleanup_errors=cleanup_errors)
        if cleanup_errors:raise RuntimeError('Cleanup incomplete; no next arm authorized')
        return record
    def run(self):
        errors=[];old={}
        for sig in (signal.SIGTERM,signal.SIGINT):old[sig]=signal.signal(sig,lambda s,f:setattr(self,'stopping',True))
        try:
            self.prepare()
            for label,arm in ARMS:
                self.check();self.run_arm(label,arm)
        except BaseException as exc:errors.append(repr(exc));self.event('coordinator_error',error=repr(exc))
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
            done=[r['arm'] for r in self.arms if r['status']=='complete_awaiting_cpu_grading']
            complete=len(done)==3 and not errors
            terminal=seal({'schema':'s2-agent-triplet-terminal-v1','status':'complete_awaiting_cpu_grading' if complete else 'partial_or_failed','config_sha256':self.config['sha256'],'registered_tasks':20,'executed_ordinal':0,'instance_id':self.config['instance_id'],'planned_arms':[x[0] for x in ARMS],'completed_arms':done,'not_started_arms':[x[0] for x in ARMS if x[0] not in [r['arm'] for r in self.arms]],'arms':self.arms,'errors':errors,'released_unix':self.budget.released if self.budget else None,'absolute_deadline_unix':self.budget.deadline if self.budget else None,'finished_unix':time.time(),'last_event_sha256':self.events[-1]['sha256'] if self.events else None,'CPU_grading_performed':False,'task_success_claim':False,'holder_policy':'OFF','holder_restoration_authorized':False})
            publish(self.output/'triplet-terminal.json',terminal);self.journal.close()
        return 0 if complete else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--ready-file',required=True);p.add_argument('--release-file',required=True);a=p.parse_args()
    config=read_json(a.config,sealed=True)
    return Coordinator(config,a.output_dir,a.ready_file,a.release_file).run()
if __name__=='__main__':raise SystemExit(main())
