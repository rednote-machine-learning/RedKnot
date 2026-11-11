"""CPU preparation of the independent, uniformly accelerated three-arm run."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

D=Path('/mnt/tidal-alsh01/dataset/redone/RedKnot_DeepInfer_20261028/s2_dspark_agent_prep_20261110.h08kqgk6')
R=D.parent
B=R/'s2_agent_prep_20261110.vifn0eiu'
B3=R/'s2_agent_retry_20261110_v3'
PY='/tmp/redknot-dsv41-sm90.6ZbGzo/di-venv/bin/python'
sys.path[:0]=[str(D),str(D/'workflow')]
from service.config import build_config,validate_config,sources
from native_decode_profile import DecodeProfile,digest,seal
import run_triplet
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    if Path(p).exists():
        assert read(p)==v, 'Existing preparation artifact differs: '+str(p)
        return
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def reference(p):
    v=read(p)
    return dict(kind='source',path=str(p),file_sha256=sha(p),seal_sha256=v['sha256'],required_fields=dict(passed=True,gpu_initialized=False))

assert not (D/'dispatch.json').exists() and not (D/'Run').exists()
old=read(B3/'triplet-config.json')
service_env=dict(old['environment']['service'])
service_env.update(DecodeProfile('dspark').environment())
service_env['PYTHONPATH']=str(D/'NativeSnapshot')+':'+str(D)
write(D/'environment.json',service_env)
cpu_env=dict(os.environ,**service_env)
cpu_env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',S2_EXPERIMENT_ROOT=str(R),
    S2_TOKENIZER='/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4.1-Flash-1017/tokenizer.json')
cpu_env['PYTHONPATH']=service_env['PYTHONPATH']+':'+str(D/'service')+':'+str(D/'GatedEngram')
plan=read(D/'cpu-test-plan.json')
test_results=[]
source_before=sources(D,R)
for i,step in enumerate(plan):
    args=[PY,'-B','-m','unittest','-v',*step['modules']]
    result=subprocess.run(args,cwd=D/step['cwd'],env=cpu_env,text=True,capture_output=True,timeout=300)
    log=D/f'focused-cpu-{i}.log';log.write_text(result.stdout+result.stderr)
    test_results.append(dict(argv=args,cwd=str(D/step['cwd']),returncode=result.returncode,log=str(log),log_sha256=sha(log)))
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
assert sources(D,R)==source_before,'Source modified during CPU proof'
proof=seal(dict(schema='s2-dspark-agent-cpu-admission-v1',passed=True,status='passed',gpu_initialized=False,
    tests=test_results,sources=source_before,observed_unix=time.time()))
write(D/'cpu-admission.json',proof)
cfg=build_config(D,R,'/mnt/tidal-alsh01/dataset/redone/checkpoints/opensource/DeepSeek-V4.1-Flash-1017',
    R/'benchmark_rag_redknot_v1/inputs-128-original/manifest.json')
validate_config(cfg)
write(D/'service-config.json',cfg)
c={k:v for k,v in old.items() if k not in ('sha256','round_initial_release_unix','absolute_deadline_unix')}
c.update(schema='s2-dspark-agent-triplet-config-v1',arm_order=['Full','Prefix','S2'],holder_policy='OFF',decode_profile='dspark',
    cwd=str(D),service_config=str(D/'service-config.json'),
    framework_script=str(D/'framework-overlay/run_official_retry_greedy.py'))
c['environment']['service']=service_env
# Reuse unchanged framework/container evidence; bind the new inference implementation separately.
c['admission_refs']=[r for r in c['admission_refs'] if Path(r['path']).name not in ('service-cpu-proof-v2.json','control-cpu-proof.json')]+[reference(D/'cpu-admission.json')]
c['source_sha256']={p:h for p,h in c['source_sha256'].items() if not (p.startswith(str(B3)) or p.startswith(str(B)+'/workflow/')) and p!=str(B/'service-config.json')}
for folder in ('workflow','framework-overlay','postrun','report'):
    for p in (D/folder).glob('*.py'):c['source_sha256'][str(p)]=sha(p)
c['source_sha256'][str(D/'service-config.json')]=sha(D/'service-config.json')
c=seal(c);run_triplet.validate_config(c);write(D/'triplet-config.json',c)
life=dict(schema='s2-dspark-agent-lifecycle-v1',timeout_seconds=7200,holder_policy='OFF',
    cwd=str(D),runner_argv=[str(D/'workflow/run_triplet.py'),'--config',str(D/'triplet-config.json')],
    output_dir=str(D/'Run'),environment_json=str(D/'environment.json'),cpu_admission=str(D/'cpu-admission.json'),
    holder_off_receipt=str(R/'s2_holder_off_user_20261110/status.json'),
    source_sha256={**c['source_sha256'],str(D/'triplet-config.json'):sha(D/'triplet-config.json'),
        str(D/'environment.json'):sha(D/'environment.json'),str(D/'cpu-admission.json'):sha(D/'cpu-admission.json')})
life=seal(life);write(D/'lifecycle-config.json',life)
r=subprocess.run([PY,'-B',str(D/'workflow/s2_lifecycle.py'),'--config',str(D/'lifecycle-config.json'),'--check-only'],cwd=D,text=True,capture_output=True,timeout=120)
(D/'lifecycle-check.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
result=seal(dict(schema='s2-dspark-agent-preparation-v1',status='cpu_ready',directory=str(D),gpu_started=False,
    service_config_sha256=cfg['sha256'],triplet_config_sha256=c['sha256'],lifecycle_config_sha256=life['sha256'],
    cpu_admission=proof,lifecycle_check=json.loads(r.stdout),holder_policy='OFF'))
write(D/'preparation.json',result)
print(json.dumps(result))
