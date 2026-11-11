"""Build a CPU-only deployment script from stable local sources."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
P=B.parent/'s2-agent-v1'
D=json.loads((B/'remote-creation.json').read_text())['directory']
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def tree(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py')) if not any(n in ('.git','__pycache__','.pytest_cache') for n in p.relative_to(root).parts)}
parent=tree(P/'GatedEngram');current=tree(B/'GatedEngram')
assert digest(parent)=='0321d533deceb984c126b07d24a099316cc7c8dcc769e470ccbcbc7095fa4d36'
changed={k:v for k,v in current.items() if parent.get(k)!=v}
assert set(parent)<=set(current)
admission=dict(schema='s2-dspark-adapted-gated-sources-v1',parent_gated_sha256=digest(parent),gated=current,changed=changed,
    native_changed=False,policy_math_change_requested=False,adapter='native DFlash tail features and graph-aware worker')
admission['sha256']=digest(admission)
files={}
for folder in ('service','redknot','workflow','framework-overlay','postrun','report'):
    for p in (B/folder).rglob('*'):
        if p.is_file() and p.suffix in ('.py','.cpp') and '__pycache__' not in p.parts:
            files[str(p.relative_to(B))]=p.read_text()
for name in ('native_decode_profile.py','dflash_tail_adapter.py','prefix_decode_contract.py','test_prefix_decode_contract.py','prepare_stage.py','cpu-test-plan.json'):
    files[name]=(B/name).read_text()
for name in changed:files['GatedEngram/'+name]=(B/'GatedEngram'/name).read_text()
files['adapted-gated-sources.json']=json.dumps(admission,ensure_ascii=False,indent=2)+'\n'
pins={k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()}
payload='from pathlib import Path\nimport json,hashlib,subprocess,os,time\nD=Path('+repr(D)+')\nfiles='+repr(files)+'\npins='+repr(pins)+'\n'
payload+='''assert not (D/'dispatch.json').exists() and not (D/'Run').exists(), 'Already dispatched'
for name,content in files.items():
 p=D/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
assert all(hashlib.sha256((D/name).read_bytes()).hexdigest()==h for name,h in pins.items())
result=dict(schema='s2-dspark-agent-deployment-v1',directory=str(D),source_sha256=pins,observed_unix=time.time(),gpu_started=False)
result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
(D/'deployment.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''
(B/'deploy_server.py').write_text(payload)
(B/'local-deployment-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
print(json.dumps(dict(files=len(files),gated_changed=list(changed),source_sha256=digest(pins))))
