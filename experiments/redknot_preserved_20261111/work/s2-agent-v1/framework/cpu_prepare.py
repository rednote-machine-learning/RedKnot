"""Offline CPU preparation; never instantiates a model, workspace, or container."""
from __future__ import annotations
import ast, hashlib, json, logging, os, pathlib, subprocess, sys, tomllib, types
B=pathlib.Path(__file__).resolve().parent;STAGE=B.parent;D=STAGE/'dataset';UP=B/'benchmarks';SDK=UP/'vendor/software-agent-sdk';EXT=B/'extensions'
sys.path.insert(0,str(B/'_deps'))
from jinja2 import Environment, FileSystemLoader
import yaml
PINS={'benchmarks':'60b3548cea8b41276ac0d44ea36a01b82936bbe4','sdk':'43376f1868ffd702746080714a59c16d3f69ec12','extensions':'4f91a480f74732b2653fdd6f1357637e5575bf91'}
def sha(b):return hashlib.sha256(b).hexdigest()
def seal(obj):obj['sha256']=sha(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode());return obj
def file_ref(p):return {'path':str(p.relative_to(STAGE)),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}
def git(p,*args):return subprocess.check_output(['git','-C',str(p),*args],text=True).strip()
def actual_functions(path,names,namespace):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names];assert {n.name for n in nodes}==set(names)
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace);return namespace

def main():
 assert sys.version_info>=(3,12)
 identities={}
 for name,p in [('benchmarks',UP),('sdk',SDK),('extensions',EXT)]:
  assert git(p,'rev-parse','HEAD')==PINS[name]
  assert not git(p,'status','--porcelain','--untracked-files=no'),f'Upstream working tree changed: {name}'
  identities[name]={'url':git(p,'remote','get-url','origin'),'commit':PINS[name],'tree':git(p,'rev-parse','HEAD^{tree}'),'tracked_worktree_clean':True}
 assert git(UP,'ls-tree','HEAD','vendor/software-agent-sdk').split()[2]==PINS['sdk']
 selection=json.loads((D/'selection-manifest.json').read_text());sbody={k:v for k,v in selection.items() if k!='sha256'};assert sha(json.dumps(sbody,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())==selection['sha256']
 for p,s in selection['files'].items():assert sha((D/p).read_bytes())==s
 rows=[json.loads(x) for x in (D/'agent_visible/tasks20.jsonl').read_text().splitlines()];assert [r['instance_id'] for r in rows]==selection['selected_instance_ids']
 allowed=set(selection['agent_visible_fields']);forbidden=set(selection['forbidden_agent_fields'])
 assert all(set(r)==allowed and not set(r)&forbidden for r in rows)
 render_src=UP/'benchmarks/swebench/run_infer.py';prompt=UP/'benchmarks/swebench/prompts/default.j2'
 ns=actual_functions(render_src,['get_instruction'],{'os':os,'Environment':Environment,'FileSystemLoader':FileSystemLoader})
 metadata=types.SimpleNamespace(details={},prompt_path=str(prompt));prompt_rows=[]
 for r in rows:
  clean=dict(r,repo_path=f"/workspace/{r['repo'].split('/')[-1]}/")
  text=ns['get_instruction'](clean,metadata,'/workspace');assert r['problem_statement'] in text and r['base_commit'] in text
  # Adversarial extra grader fields cannot influence the actual official template.
  canary='S2_GRADER_ONLY_CANARY_DO_NOT_SEND';aug=dict(clean,**{k:canary for k in forbidden})
  assert ns['get_instruction'](aug,metadata,'/workspace')==text and canary not in text
  p=D/'agent_visible/prompts'/f"{r['instance_id']}.txt";p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_text()==text
  else:p.write_text(text)
  prompt_rows.append({'instance_id':r['instance_id'],'prompt_sha256':sha(text.encode()),'prompt_bytes':len(text.encode()),'repo_path':clean['repo_path'],'original_problem_statement_sha256':sha(r['problem_statement'].encode())})
 # Exercise official skills file-discovery functions without importing SDK/model modules.
 util=SDK/'openhands-sdk/openhands/sdk/skills/utils.py'
 sns=actual_functions(util,['find_skill_md','find_skill_md_directories','find_regular_md_files'],{'Path':pathlib.Path})
 assert not (EXT/'marketplaces/default.json').exists()
 skill_files=sns['find_skill_md_directories'](EXT/'skills');skill_files+=sns['find_regular_md_files'](EXT/'skills',{p.parent for p in skill_files});skills=[]
 for p in sorted(skill_files):
  text=p.read_text();fm={}
  if text.startswith('---\n') and '\n---' in text[4:]:fm=yaml.safe_load(text.split('---',2)[1]) or {}
  skills.append({**file_ref(p),'declared_name':fm.get('name'),'frontmatter_keys':sorted(fm),'declared_triggers':fm.get('triggers'),'disable_model_invocation':fm.get('disable-model-invocation')})
 # Verify actual official CLI and delegation statements are present; no synthetic roles.
 infer=render_src.read_text();assert 'if self.metadata.enable_delegation:' in infer and 'tools.append(Tool(name=TaskToolSet.name))' in infer
 conf=tomllib.loads((UP/'pyproject.toml').read_text());sdkconf=tomllib.loads((SDK/'openhands-sdk/pyproject.toml').read_text())
 src_paths=[UP/'pyproject.toml',UP/'uv.lock',SDK/'pyproject.toml',SDK/'uv.lock',render_src,prompt,UP/'benchmarks/utils/args_parser.py',UP/'benchmarks/utils/agent_context.py',UP/'benchmarks/utils/dataset.py',UP/'benchmarks/swebench/config.py',SDK/'openhands-sdk/openhands/sdk/skills/skill.py',util,SDK/'openhands-workspace/openhands/workspace/apptainer/workspace.py']
 manifest=seal({'schema':'s2-openhands-framework-pins-v1','sources':identities,'source_files':[file_ref(p) for p in src_paths],'host_python_requires':conf['project']['requires-python'],'sdk_version':sdkconf['project']['version'],'sdk_python_requires':sdkconf['project']['requires-python'],'dependency_install':['uv','sync','--frozen'],'dependency_notes':['Upstream local Apptainer source-minimal image recipe installs managed Python3.13 inside the container; host benchmark requires>=3.12.','Actual full SDK imports/container launch are not validated by this offline AST/template dry run.'],'runtime_contract':{'agent_type':'default','tool_preset':'default','enable_delegation':True,'delegation_mechanism':'official TaskToolSet, model decides whether/how to delegate; no forced roles','enable_browser':False,'enable_condenser':True,'condenser_max_size':240,'condenser_keep_first':2,'max_iterations':500,'max_retries':3,'num_workers':1,'n_limit':0,'critic':'official default, inspect actual runtime configuration before admission','prompt_file':str(prompt.relative_to(STAGE)),'dataset_file':'dataset/agent_visible/tasks20.jsonl','selected_ids_file':'dataset/selected_instances.txt','llm_config':'root-owned Full/S2 endpoint configuration; identical model/decoding/tool budgets required','environment':{'EXTENSIONS_REF':PINS['extensions']},'workspace':'isolated official Docker or Apptainer, pending actual server CPU qualification; LocalWorkspace prohibited','container_gpu':False},'public_skills':{'repository':identities['extensions'],'default_marketplace_path':'marketplaces/default.json','default_marketplace_exists':False,'actual_official_fallback':'scan skills/SKILL.md directories plus legacy standalone md; child reference md excluded','discovered_files':skills,'status':'official file discovery executed, not equivalent to actual SDK Skill.load success or actual model-visible skill contents'},'selection_manifest_sha256':selection['sha256'],'grader_contract':{'official_package':'swebench==4.1.0','dataset':'dataset/grader_only/selected20.jsonl','must_not_mount':['dataset/grader_only','framework/packages','framework/evaluator-source'],'no_gold_patch_or_test_patch_in_model_cache':True,'arm_runs_must_use_distinct_run_id':True},'pending_runtime_checks':['isolated container CPU execution and filesystem boundaries','resolved immutable base-image/agent-image digests and per-task base commit','full SDK imports, actual public skill load manifest, recorded model request construction','endpoint/native tool-call formatting and same policy qualification','root-owned GPU lifecycle authorization/release']})
 (B/'framework-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 proof=seal({'schema':'s2-offline-cpu-preparation-v1','status':'passed_metadata_prompt_and_boundary_checks_only','framework_manifest_sha256':manifest['sha256'],'selection_manifest_sha256':selection['sha256'],'checks':{'upstream_git_pins_and_clean_worktrees':True,'exact_sdk_submodule':True,'candidate_rows':500,'selected_rows':20,'selected_repositories':12,'task_ids_and_file_shas_verified':True,'official_get_instruction_ast_executed':True,'default_prompt_preserved':True,'grader_field_canary_cannot_affect_prompt':True,'all_agent_fields_whitelisted':True,'skills_discovery_actual_functions_executed':True,'skills_files_discovered':len(skills),'models_initialized':0,'container_workloads_started':0,'gpu_operations':0,'network_calls_during_dryrun':0},'prompts':prompt_rows,'limits':'Not a full SDK import, container, model-serving, task-success or GPU qualification proof.'})
 (B/'cpu-preparation.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'framework_seal':manifest['sha256'],'cpu_seal':proof['sha256'],'skills_discovered':len(skills),'prompt_checks':len(prompt_rows)},indent=2))
if __name__=='__main__':main()
