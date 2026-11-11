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
    # Bind before any evaluate_instance request; cannot contact an occupied GPU endpoint.
    server=http.server.ThreadingHTTPServer(('127.0.0.1',8123),Handler)
    Handler.requests=[]
    from routing_guard import install
    LLM=install()
    from benchmarks.swebench import run_infer
    SWEBenchEvaluation=run_infer.SWEBenchEvaluation
    from benchmarks.utils.models import EvalMetadata,EvalInstance
    from openhands.sdk.critic import AgentFinishedCritic
    dest=pathlib.Path(directory)/'initial-request-cpu-capture';dest.mkdir(exist_ok=False)
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
    if len(Handler.requests)!=1:raise ValueError('Expected exactly one initial mock request; got '+str(len(Handler.requests)))
    request=Handler.requests[0]
    if request['path']!='/v1/chat/completions':raise ValueError('Unexpected route')
    if not request['body'].get('tools'):raise ValueError('Missing actual official tools schema')
    patch=verify_mock_patch(workspace,evaluator.get_repo_path(instance),row,admission,result,dest)
    report={'schema':'s2-official-initial-request-cpu-capture-v1','instance_id':row['instance_id'],'ordinal':ordinal,'base_commit':row['base_commit'],'base_environment_proof':admission['proof_ref'] if admission else None,'patch_baseline_admission_sha256':admission['sha256'] if admission else None,'mock_patch':patch,'official_evaluate_instance_called':True,'mock_finish_response':True,'formal_result':False,'gpu_calls':False,'request':request,'selection_sha256':hashlib.sha256((B.parent/'dataset/selection-manifest.json').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((B/'config/llm.Full.json').read_bytes()).hexdigest(),'interpretation':'Actual first request messages/tools; token counts require target checkpoint encoding. Mock response does not measure model behavior, delegation, or task success.'}
    target=dest/'request.json';target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return str(target)
