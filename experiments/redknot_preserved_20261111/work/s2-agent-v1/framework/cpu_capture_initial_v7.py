"""Capture one official initial OpenHands request against a CPU mock responder.

Called only from cpu_workspace_probe after actual isolation checks. Uses the
registered task, base image, unmodified evaluate_instance, official prompt/tools,
500-iteration policy and default critic; mock immediately calls finish. Never a
formal task outcome, benchmark arm, or inference/numerical qualification.
"""
from __future__ import annotations
import hashlib,http.server,json,pathlib,shlex,sys,threading
B=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(B/'benchmarks'))
from cpu_sdk_wire_probe import Handler

class RequestJournal(list):
    """Persist each parsed actual request before the mock sends its response."""
    def __init__(self,path,instance_id):
        super().__init__();self.path=pathlib.Path(path);self.instance_id=instance_id;self.lock=threading.Lock()
    def append(self,request):
        with self.lock:
            super().append(request)
            value={'schema':'s2-official-initial-all-mock-requests-v1','instance_id':self.instance_id,'request_count':len(self),'requests':list(self),'formal_result':False,'gpu_or_real_model_calls':False}
            data=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
            pending=self.path.with_suffix('.tmp');pending.write_text(data);pending.replace(self.path)

def select_task_request(requests,row):
    candidates=[];descriptions=[]
    issue=row['problem_statement']
    if not isinstance(issue,str) or not issue:raise ValueError('Original task issue is missing')
    for ordinal,request in enumerate(requests):
        body=request.get('body',{});texts=[]
        for message in body.get('messages',[]):
            content=message.get('content')
            if isinstance(content,str):texts.append(content)
            elif isinstance(content,list):texts.extend(b['text'] for b in content if isinstance(b,dict) and b.get('type')=='text' and isinstance(b.get('text'),str))
        has_tools=bool(body.get('tools'));has_issue=any(issue in text for text in texts)
        if request.get('path')=='/v1/chat/completions' and has_tools and has_issue:candidates.append(ordinal)
        descriptions.append({'ordinal':ordinal,'path':request.get('path'),'has_tools':has_tools,'contains_exact_original_issue':has_issue,'body_sha256':request.get('body_sha256'),'classification':'unknown_extra_diagnostic_request'})
    if len(candidates)!=1:raise ValueError('Expected one unambiguous tool-enabled original-task request; candidates='+repr(candidates))
    chosen=candidates[0];descriptions[chosen]['classification']='original_task_tool_request'
    return requests[chosen],chosen,descriptions

def verify_mock_patch(workspace, repo_path, row, admission, result, directory):
    """A no-op mock may retain only the witnessed official setup delta."""
    def git(*argv):
        command='git -C '+shlex.quote(repo_path)+' '+shlex.join(argv)
        value=workspace.execute_command(command)
        if value.exit_code!=0:raise ValueError('CPU mock repository inspection failed')
        return value.stdout
    dataset=row['base_commit']
    baseline=admission['environment_commit'] if admission else dataset
    raw=git('--no-pager','diff','--no-color',dataset,'HEAD')
    canonical=git('--no-pager','diff','--no-color',baseline,'HEAD')
    expected=admission['setup_delta_sha256'] if admission else hashlib.sha256(b'').hexdigest()
    if hashlib.sha256(raw.encode()).hexdigest()!=expected:
        raise ValueError('CPU mock raw patch differs from admitted official setup delta')
    if canonical or result.test_result.get('git_patch'):
        raise ValueError('CPU mock unexpectedly changed the admitted environment tree')
    dest=pathlib.Path(directory)
    rawpath=dest/'mock.dataset-base.patch';rawpath.write_text(raw)
    canonicalpath=dest/'mock.environment-base.patch';canonicalpath.write_text(canonical)
    return {'dataset_base_commit':dataset,'environment_commit':baseline,'raw_patch_path':str(rawpath),'raw_patch_sha256':hashlib.sha256(raw.encode()).hexdigest(),'model_patch_path':str(canonicalpath),'model_patch_sha256':hashlib.sha256(canonical.encode()).hexdigest(),'canonical_patch_empty':True,'raw_contains_only_admitted_official_setup_delta':True,'formal_result':False,'scored':False}

def capture(workspace, ordinal, directory, base_environment_proof=None):
    # Official version/build helpers consult Git at import time as well as run
    # time. Keep the real pinned repository cwd for both, and restore afterward.
    from run_official import official_repository_cwd
    with official_repository_cwd():
        return _capture(workspace,ordinal,directory,base_environment_proof)

def _capture(workspace, ordinal, directory, base_environment_proof=None):
    ids=(B.parent/'dataset/selected_instances.txt').read_text().splitlines()
    if ordinal not in range(20):raise ValueError('Ordinal outside sealed20')
    rows=[json.loads(x) for x in (B.parent/'dataset/agent_visible/tasks20.jsonl').read_text().splitlines()]
    row=rows[ordinal]
    if row['instance_id']!=ids[ordinal]:raise ValueError('Selection order mismatch')
    admission=None
    if base_environment_proof:
        import patch_baseline
        selected=json.loads((B.parent/'dataset/selection-manifest.json').read_text())['cases'][ordinal]
        admission=patch_baseline.load_proof(base_environment_proof,selected)
    expected_head=admission['environment_commit'] if admission else row['base_commit']
    base=workspace.execute_command('git -C /testbed rev-parse HEAD')
    if base.exit_code!=0 or base.stdout.strip()!=expected_head:
        raise ValueError('Actual SIF /testbed commit differs; official setup commit requires an admitted --base-environment-proof')
    if admission:
        tree=workspace.execute_command('git -C /testbed rev-parse HEAD^{tree}')
        if tree.exit_code!=0 or tree.stdout.strip()!=admission['environment_tree']:
            raise ValueError('Actual SIF /testbed tree differs from admitted official environment')
    dest=pathlib.Path(directory)/'initial-request-cpu-capture';dest.mkdir(exist_ok=False)
    journal=RequestJournal(dest/'requests.all.json',row['instance_id'])
    CaptureHandler=type('CaptureHandler',(Handler,),{'requests':journal})
    # Bind before any evaluate_instance request; cannot contact an occupied GPU endpoint.
    server=http.server.ThreadingHTTPServer(('127.0.0.1',8123),CaptureHandler)
    from routing_guard import install
    LLM=install()
    from benchmarks.swebench import run_infer
    SWEBenchEvaluation=run_infer.SWEBenchEvaluation
    from benchmarks.utils.models import EvalMetadata,EvalInstance
    from openhands.sdk.critic import AgentFinishedCritic
    # Match run_infer.main: get_instruction requires the initialized empty dict.
    metadata=EvalMetadata(llm=LLM.model_validate_json((B/'config/llm.Full.json').read_text()),dataset=str(B.parent/'dataset/agent_visible/tasks20.jsonl'),dataset_split='test',max_iterations=500,eval_output_dir=str(dest),details={},prompt_path=str(B/'benchmarks/benchmarks/swebench/prompts/default.j2'),critic=AgentFinishedCritic(),workspace_type='apptainer',enable_delegation=True,enable_condenser=True,condenser_max_size=240,condenser_keep_first=2,tool_preset='default',agent_type='default')
    evaluator=SWEBenchEvaluation(metadata=metadata,num_workers=1)
    patch_handle=patch_baseline.install(run_infer,{row['instance_id']:admission},dest) if admission else None
    threading.Thread(target=server.serve_forever,daemon=True).start()
    instance=EvalInstance(id=row['instance_id'],data=dict(row))
    try:
        result=evaluator.evaluate_instance(instance,workspace)
    finally:
        server.shutdown();server.server_close()
        if patch_handle is not None:patch_baseline.restore(run_infer,patch_handle)
    request,request_ordinal,request_descriptions=select_task_request(journal,row)
    if request['path']!='/v1/chat/completions':raise ValueError('Unexpected route')
    if not request['body'].get('tools'):raise ValueError('Missing actual official tools schema')
    patch=verify_mock_patch(workspace,evaluator.get_repo_path(instance),row,admission,result,dest)
    report={'schema':'s2-official-initial-request-cpu-capture-v1','instance_id':row['instance_id'],'ordinal':ordinal,'base_commit':row['base_commit'],'base_environment_proof':admission['proof_ref'] if admission else None,'patch_baseline_admission_sha256':admission['sha256'] if admission else None,'mock_patch':patch,'official_evaluate_instance_called':True,'mock_finish_response':True,'formal_result':False,'gpu_calls':False,'request':request,'selection_sha256':hashlib.sha256((B.parent/'dataset/selection-manifest.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((B/'config/llm.Full.json').read_bytes()).hexdigest(),'interpretation':'Actual first request messages/tools; token counts require target checkpoint encoding. Mock response does not measure model behavior, delegation, or task success.'}
    report.update(actual_request_count=len(journal),selected_task_request_ordinal=request_ordinal,request_descriptions=request_descriptions,all_requests_ref={'path':str(journal.path),'file_sha256':hashlib.sha256(journal.path.read_bytes()).hexdigest()},extra_request_interpretation='All additional mock requests remain preserved. Pinned official AutoTitleSubscriber can issue a no-tools title completion in the background; extras are not classified as title requests without inspecting their actual payloads. Future real service requests remain included in accounting.')
    target=dest/'request.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return str(target)
