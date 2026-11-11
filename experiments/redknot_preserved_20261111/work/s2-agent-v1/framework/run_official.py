"""External S2 adapter around the unmodified official SWE-bench entry point.

Default action prints a plan. --run requires the separately admitted endpoint and
container environment. The only workspace adaptation sets the existing public
ApptainerWorkspace.use_fakeroot=False option when already running as real root.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, pathlib, signal, subprocess, sys
B=pathlib.Path(__file__).resolve().parent;STAGE=B.parent
sys.path.insert(0,str(B))
from runtime_policy import admitted_environment, validate_config, workspace_kwargs
import workspace_ownership
import admitted_images
import patch_baseline

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_inputs():
 m=json.loads((B/'framework-manifest.json').read_text());s=json.loads((STAGE/'dataset/selection-manifest.json').read_text())
 for name,path in [('benchmarks',B/'benchmarks'),('sdk',B/'benchmarks/vendor/software-agent-sdk'),('extensions',B/'extensions')]:
  actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
  if actual!=m['sources'][name]['commit']:raise ValueError(f'{name} commit differs from pin')
 for f in m['source_files']:
  if sha(STAGE/f['path'])!=f['sha256']:raise ValueError(f'Source changed: {f["path"]}')
 for p in ['agent_visible/tasks20.jsonl','selected_instances.txt']:
  if sha(STAGE/'dataset'/p)!=s['files'][p]:raise ValueError(f'Input changed: {p}')
 rows=[json.loads(x) for x in (STAGE/'dataset/agent_visible/tasks20.jsonl').read_text().splitlines()]
 if any(set(r)!=set(s['agent_visible_fields']) for r in rows):raise ValueError('Unsafe/nonwhitelisted task fields')
 return m,s

def build_plan(args):
 m,s=verify_inputs();llm=pathlib.Path(args.llm_config).resolve();out=pathlib.Path(args.output_dir).resolve()
 # Full, Prefix and S2 use distinct directories; the gateway process selects the
 # actual KV policy. SDK caching_prompt is not a native prefix-cache switch.
 out=out/args.arm
 argv=['swebench-infer',str(llm),'--dataset',str(STAGE/'dataset/agent_visible/tasks20.jsonl'),'--split','test','--select',str(STAGE/'dataset/selected_instances.txt'),'--workspace',args.workspace,'--num-workers','1','--n-limit','0','--max-iterations','500','--max-retries','3','--tool-preset','default','--agent-type','default','--enable-delegation','--enable-condenser','--condenser-max-size','240','--condenser-keep-first','2','--prompt-path',str(B/'benchmarks/benchmarks/swebench/prompts/default.j2'),'--output-dir',str(out),'--note','S2-'+args.arm]
 ids=(STAGE/'dataset/selected_instances.txt').read_text().splitlines()
 ordinal=getattr(args,'ordinal',None)
 if ordinal is not None:
  if ordinal not in range(len(ids)):raise ValueError('Ordinal outside the sealed registered20')
  subset=B/'config/subsets'/f'{ordinal:02}.txt'
  if subset.read_text()!=ids[ordinal]+'\n':raise ValueError('Ordinal file does not match sealed selection')
  argv[argv.index('--select')+1]=str(subset)
  out=out/f'ordinal{ordinal:02}'
  argv[argv.index('--output-dir')+1]=str(out)
 return {'schema':'s2-official-agent-launch-plan-v2','upstream_entrypoint':'benchmarks.swebench.run_infer.main','argv':argv,'environment':{'EXTENSIONS_REF':m['sources']['extensions']['commit']},'arm':args.arm,'workspace':args.workspace,'apptainer_real_root_without_fakeroot':bool(args.apptainer_real_root),'registered_instances':ids,'execution_instances':ids if ordinal is None else [ids[ordinal]],'ordinal':ordinal,'selection_sha256':s['sha256'],'framework_sha256':m['sha256'],'custom_prompt_or_roles':False,'conversation_timeout_seconds':1800,'hard_task_timeout_owner':'external lifecycle, not SDK fake-response timeout','status':'plan_only' if not args.run else 'caller_requested_execution','requirements':['root-owned endpoint admission and GPU lifecycle release before any model calls','actual isolated container CPU proof; fixed image digests/base commit','actual public-skills cache prewarmed from pinned local clone; no GPU-run dependency on preparation tunnel']}

def install_apptainer_root_option(module,real_uid):
 if real_uid!=0:raise RuntimeError('use_fakeroot=False adapter is restricted to an already real-root process')
 original=module.ApptainerWorkspace
 module.ApptainerWorkspace=workspace_ownership.make_workspace_class(original,lambda kwargs:workspace_kwargs(kwargs,real_uid))
 return original

@contextlib.contextmanager
def official_repository_cwd():
 previous=pathlib.Path.cwd()
 os.chdir(B/'benchmarks')
 try:yield
 finally:os.chdir(previous)

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--llm-config',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--arm',choices=['Full','Prefix','S2'],required=True);p.add_argument('--workspace',choices=['docker','apptainer'],required=True);p.add_argument('--apptainer-real-root',action='store_true');p.add_argument('--ordinal',type=int);p.add_argument('--image-provenance',action='append',default=[]);p.add_argument('--run',action='store_true');args=p.parse_args(argv)
 if args.apptainer_real_root and args.workspace!='apptainer':p.error('--apptainer-real-root requires --workspace apptainer')
 plan=build_plan(args)
 if not args.run:print(json.dumps(plan,indent=2));return
 if args.workspace!='apptainer' or not args.apptainer_real_root:raise RuntimeError('Current admitted runtime requires real-root isolated Apptainer')
 if args.apptainer_real_root and os.getuid()!=0:raise RuntimeError('Root-mode option requires real uid0')
 validate_config(args.llm_config)
 m,s=verify_inputs()
 rows=[r for r in s['cases'] if r['instance_id'] in plan['execution_instances']]
 images=admitted_images.load_images(args.image_provenance,rows,m['sources']['sdk']['commit'])
 plan['admitted_images']={k:{field:v[field] for field in ('sif_path','sif_sha256','base_commit','provenance_path','provenance_file_sha256')} for k,v in images.items()}
 for k,v in plan['environment'].items():
  if os.getenv(k) not in (None,v):raise RuntimeError(f'Conflicting pinned {k}')
  os.environ[k]=v
 if not pathlib.Path(args.llm_config).is_file():raise FileNotFoundError(args.llm_config)
 sys.path.insert(0,str(B/'benchmarks'))
 # EXTENSIONS_REF is read at SDK import, so import only after setting its pin.
 # Official version helpers also invoke git during import, so both import and
 # main execute from the already-pinned benchmark repository.
 with admitted_environment() as runtime_receipt, official_repository_cwd():
  from routing_guard import install
  install()
  from benchmarks.swebench import run_infer
  original=install_apptainer_root_option(run_infer,os.getuid())
  image_originals=admitted_images.install(run_infer,images)
  receipt_dir=pathlib.Path(plan['argv'][plan['argv'].index('--output-dir')+1]);receipt_dir.mkdir(parents=True,exist_ok=True)
  workspace_ownership.configure(receipt_dir)
  baseline_admissions={}
  for row in rows:
   ref=images[row['instance_id']].get('base_environment_admission')
   if not isinstance(ref,dict):raise ValueError('Admitted image requires base_environment_admission reference')
   baseline_admissions[row['instance_id']]=patch_baseline.load_proof(ref['path'],row,ref=ref)
  receipt={**runtime_receipt,'plan':plan,'llm_config_sha256':sha(pathlib.Path(plan['argv'][1]))}
  # Exclusive receipt prevents accidental official result-resume reuse.
  with (receipt_dir/'s2-runtime-admission.json').open('x') as f:json.dump(receipt,f,indent=2)
  old_argv=sys.argv;sys.argv=plan['argv']
  previous_handler=signal.getsignal(signal.SIGTERM)
  def terminate(signum,frame):
   workspace_ownership.cleanup_all()
   raise SystemExit(128+signum)
  signal.signal(signal.SIGTERM,terminate)
  patch_originals=patch_baseline.install(run_infer,baseline_admissions,receipt_dir)
  try:run_infer.main()
  finally:
   workspace_ownership.cleanup_all()
   signal.signal(signal.SIGTERM,previous_handler)
   sys.argv=old_argv
   run_infer.ApptainerWorkspace=original
   admitted_images.restore(run_infer,image_originals)
   patch_baseline.restore(run_infer,patch_originals)
if __name__=='__main__':main()
