"""CPU-only actual SDK skill preload from the pinned public local source clone."""
import hashlib,json,os,pathlib,subprocess,sys
B=pathlib.Path(__file__).resolve().parent;m=json.loads((B/'framework-manifest.json').read_text());pin=m['sources']['extensions']['commit'];clone=B/'extensions'
assert subprocess.check_output(['git','-C',str(clone),'rev-parse','HEAD'],text=True).strip()==pin
os.environ['EXTENSIONS_REF']=pin
# Per-process Git rewrite, not user/global Git config: clone/fetch the exact public
# source already downloaded. SDK cache metadata still uses the official repo URL.
if os.environ.get('GIT_CONFIG_COUNT') not in (None,'0'):raise RuntimeError('Existing per-process Git config must be reviewed before skill preload')
os.environ['GIT_CONFIG_COUNT']='1';os.environ['GIT_CONFIG_KEY_0']='url.'+clone.as_uri()+'.insteadOf';os.environ['GIT_CONFIG_VALUE_0']='https://github.com/OpenHands/extensions'
from openhands.sdk.skills.skill import load_public_skills
skills=load_public_skills(ref=pin)
if not skills:raise RuntimeError('Pinned public skills did not load; do not run empty-context fallback')
rows=[]
for skill in skills:
 data=skill.model_dump(mode='json');blob=json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode();rows.append({'name':skill.name,'serialized_sha256':hashlib.sha256(blob).hexdigest()})
report={'schema':'s2-actual-sdk-public-skills-preload-v1','extensions_commit':pin,'loaded_skills':len(rows),'skills':sorted(rows,key=lambda x:x['name']),'model_calls':0,'gpu_calls':0};report['sha256']=hashlib.sha256(json.dumps(report,sort_keys=True,separators=(',',':')).encode()).hexdigest();print(json.dumps(report,indent=2))
