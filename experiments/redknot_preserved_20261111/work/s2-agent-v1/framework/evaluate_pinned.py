"""Use the official evaluator on pinned grader-only records, never an HF moving ref."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,sys
B=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(B))
from run_official import verify_inputs

def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--arm',choices=['Full','Prefix','S2'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--output-root',required=True);p.add_argument('--workspace',choices=['docker','apptainer'],required=True);p.add_argument('--run',action='store_true');a=p.parse_args();m,s=verify_inputs()
 grade=B.parent/'dataset/grader_only/selected20.jsonl'
 if hashlib.sha256(grade.read_bytes()).hexdigest()!=s['files']['grader_only/selected20.jsonl']:raise ValueError('Grader data changed')
 root=pathlib.Path(a.output_root).resolve()/a.arm/a.run_id
 plan={'schema':'s2-official-pinned-evaluator-plan-v1','dataset':str(grade),'predictions':str(pathlib.Path(a.predictions).resolve()),'output_root':str(root),'run_id':a.arm+'-'+a.run_id,'workspace':a.workspace,'official_package':'swebench==4.1.0','timeout_seconds':3600,'workers':1,'status':'plan_only' if not a.run else 'caller_requested_evaluation','grading_math':'official SWE-bench TestSpec and get_eval_report; no custom success metric','model_calls':0}
 if not a.run:print(json.dumps(plan,indent=2));return
 # Refuse cached scores or writable sandboxes from another evaluation.
 root.mkdir(parents=True,exist_ok=False)
 predictions=[json.loads(x) for x in pathlib.Path(a.predictions).read_text().splitlines() if x.strip()];ids=[x['instance_id'] for x in predictions]
 if len(ids)!=len(set(ids)) or not set(ids)<=set(s['selected_instance_ids']):raise ValueError('Duplicate/outside selected20 predictions')
 if any(set(x)-{'instance_id','model_name_or_path','model_patch'} for x in predictions):raise ValueError('Expected official prediction format only')
 sys.path.insert(0,str(B/'benchmarks'))
 import importlib.metadata
 if importlib.metadata.version('swebench')!='4.1.0':raise RuntimeError('Official evaluator package pin mismatch')
 if a.workspace=='docker':
  from swebench.harness.run_evaluation import main as official
  official(dataset_name=str(grade),split='test',instance_ids=ids,predictions_path=plan['predictions'],max_workers=1,force_rebuild=False,cache_level='instance',clean=False,open_file_limit=4096,run_id=plan['run_id'],timeout=3600,namespace='swebench',rewrite_reports=False,modal=False,report_dir=str(root))
 else:
  from benchmarks.swebench.apptainer_eval import score_instance
  records={x['instance_id']:x for x in map(json.loads,grade.read_text().splitlines())};results={}
  for name in ['scores','sandboxes','cache']:(root/name).mkdir()
  for pred in predictions:
   result=score_instance(instance=records[pred['instance_id']],prediction=pred,score_dir=root/'scores',sandbox_root=root/'sandboxes',timeout_seconds=3600,apptainer_cache=root/'cache');results.update(result)
  (root/'reports.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(plan|{'status':'official_evaluation_returned'},indent=2))
if __name__=='__main__':main()
