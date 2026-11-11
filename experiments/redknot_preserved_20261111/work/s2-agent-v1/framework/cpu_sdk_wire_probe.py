"""Actual SDK/LiteLLM -> local mock HTTP wire check. CPU only; refuses occupied8123.

Run in the admitted host venv AND agent-server venv before any GPU endpoint exists.
A mock is deliberately not an inference/agent-success or numerical qualification.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, http.server, importlib, json, os, pathlib, sys, threading, time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID
B=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(B/'runtime_guard'))

class Handler(http.server.BaseHTTPRequestHandler):
    requests=[]
    def do_POST(self):
        raw=self.rfile.read(int(self.headers['Content-Length']))
        value=json.loads(raw)
        self.requests.append({'path':self.path,'body':value,'body_sha256':hashlib.sha256(raw).hexdigest(),'informational_headers':{k:self.headers[k] for k in self.headers if k.lower() in ('x-litellm-session-id','http-referer','x-title')}})
        msg={'role':'assistant','content':'CPU_WIRE_OK'}
        if value.get('tools'):
            msg.update(content=None,tool_calls=[{'id':'cpu_finish','type':'function','function':{'name':'finish','arguments':'{"message":"CPU_WIRE_OK"}'}}])
        result={'id':'cpu-mock-'+str(len(self.requests)),'object':'chat.completion','created':int(time.time()),'model':'deepseek-v4.1-flash','choices':[{'index':0,'message':msg,'finish_reason':'tool_calls' if value.get('tools') else 'stop'}],'usage':{'prompt_tokens':8,'completion_tokens':4,'total_tokens':12}}
        data=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    # Bind before SDK import/calls: if a real endpoint is already live, abort.
    server=http.server.ThreadingHTTPServer(('127.0.0.1',8123),Handler)
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
    from routing_guard import install,validate_llm
    LLM=install()
    router_name='openhands.agent_server.conversation_router'
    router_import_fresh=router_name not in sys.modules
    router=importlib.import_module(router_name)
    assert router.START_CONVERSATION_EXAMPLES
    from openhands.sdk import Agent
    from openhands.sdk.llm import Message,TextContent
    from openhands.sdk.tool.builtins.finish import FinishTool
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from openhands.tools.preset.default import register_builtins_agents
    from openhands.tools.task.manager import TaskManager
    from openhands.sdk.subagent.registry import get_agent_factory
    from openhands.sdk.critic import AgentFinishedCritic
    from openhands.sdk.agent.utils import amake_llm_completion
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    llm=LLM.model_validate_json(pathlib.Path(args.config).read_text())
    validate_llm(llm)
    assert llm.effective_max_output_tokens is None
    register_builtins_agents(enable_browser=False)
    # Match the actual conversation's metadata creation. Condenser is created
    # before the parent cache key is pinned; delegated LLMs inherit that key.
    condenser=LLMSummarizingCondenser(llm=llm.model_copy(update={'usage_id':'condenser'}),max_size=240,keep_first=2)
    session_id=UUID('6d7ef556-bf6c-47f0-ae73-9cb569b60c78')
    owner=SimpleNamespace(_state=SimpleNamespace(id=session_id),agent=SimpleNamespace(llm=llm))
    LocalConversation._pin_prompt_cache_key(owner)
    parent=SimpleNamespace(parent_conversation=SimpleNamespace(agent=Agent(llm=llm,tools=[])))
    delegated=TaskManager._get_sub_agent_from_factory(parent,get_agent_factory('general-purpose')).llm
    for target in (llm,delegated,condenser.llm):LocalConversation._pin_session_affinity_header(owner,target)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    routes=[]
    try:
        async def actual_routes():
            for name,target in [('main',llm),('delegation_general_purpose',delegated),('condenser',condenser.llm)]:
                validate_llm(target)
                messages=[Message(role='user',content=[TextContent(text='CPU route '+name)])]
                if name=='condenser':await target.acompletion(messages=messages)
                else:await amake_llm_completion(target,messages,tools=FinishTool.create())
                routes.append({'route':name,'effective_max_output_tokens':target.effective_max_output_tokens,'transport_received':True,'official_async_agent_helper':name!='condenser','session_affinity_header':target.extra_headers['x-litellm-session-id'],'official_expected_headers':{**target._openrouter_headers(),**(target.extra_headers or {})},'prompt_cache_key':target._prompt_cache_key})
        asyncio.run(actual_routes())
        before=len(Handler.requests)
        assert not AgentFinishedCritic().evaluate([],git_patch='').success
        assert len(Handler.requests)==before
        rejected=[]
        for update in [{'base_url':'https://invalid.example/v1'},{'max_output_tokens':32},{'temperature':1}]:
            try:
                llm.model_copy(update=update).completion(messages=[Message(role='user',content=[TextContent(text='must reject')])])
            except ValueError:rejected.append(next(iter(update)))
            else:raise AssertionError('Unadmitted LLM update escaped guard')
        assert len(Handler.requests)==before
        # The actual router example can be constructed, but cannot dispatch.
        # Patch the exact SDK HTTP transport call site; rejection must occur
        # before it is invoked, not merely before our local mock receives data.
        example=router.START_CONVERSATION_EXAMPLES[0]['agent']['llm']
        placeholder=LLM(model=example['model'],usage_id=example['usage_id'])
        sdk_module=importlib.import_module('openhands.sdk.llm.llm')
        with patch.object(sdk_module,'litellm_completion',side_effect=AssertionError('Blocked placeholder reached HTTP transport')) as transport:
            try:
                placeholder.completion(messages=[Message(role='user',content=[TextContent(text='must reject inert documentation example')])])
            except ValueError as exc:
                assert str(exc).startswith('S2 LLM policy mismatch:'),str(exc)
            else:raise AssertionError('Documentation placeholder escaped dispatch policy')
            assert transport.call_count==0
        inferred_cap=llm.model_copy(deep=True)
        inferred_cap._effective_max_output_tokens=32
        try:
            inferred_cap.completion(messages=[Message(role='user',content=[TextContent(text='must reject inferred cap')])])
        except ValueError as exc:
            assert 'SDK inferred an artificial output cap' in str(exc),str(exc)
        else:raise AssertionError('Inferred output cap escaped dispatch policy')
        assert len(Handler.requests)==before
    finally:
        server.shutdown();server.server_close()
    for entry,route in zip(Handler.requests,routes,strict=True):
        v=entry['body'];assert entry['path']=='/v1/chat/completions';assert v['model']=='deepseek-v4.1-flash'
        assert v['temperature']==0 and v['top_p']==1 and v['seed']==0
        assert not any(k in v for k in ('max_tokens','max_completion_tokens','max_output_tokens','stop'))
        assert v.get('tool_choice') in (None,'auto')
        assert {k.lower():value for k,value in entry['informational_headers'].items()}=={k.lower():value for k,value in route['official_expected_headers'].items()}
    assert Handler.requests[0]['body'].get('prompt_cache_key')==str(session_id)
    assert Handler.requests[1]['body'].get('prompt_cache_key')==str(session_id)
    assert len(Handler.requests)==3
    result={'schema':'s2-actual-sdk-wire-cpu-probe-v1','status':'passed','gpu_or_real_model_calls':False,'official_sdk':__import__('importlib.metadata',fromlist=['version']).version('openhands-sdk'),'config_sha256':hashlib.sha256(pathlib.Path(args.config).read_bytes()).hexdigest(),'routes':routes,'critic_llm_calls':0,'rejected_overrides':rejected,'official_router_import_under_guard':{'passed':True,'fresh_import':router_import_fresh,'documentation_examples':len(router.START_CONVERSATION_EXAMPLES)},'placeholder_constructed_but_completion_rejected_before_transport':True,'inferred_output_cap_rejected_at_dispatch':True,'official_session_metadata_preserved_on_wire':True,'prompt_cache_key_interpretation':'Official SDK informational affinity key preserved verbatim; the fixed-arm DeepInfer gateway ignores it. This does not enable, measure or prove native prefix cache reuse.','requests':Handler.requests,'limits':'Probe checks transport/API, not real model encoding, natural delegation frequency, or task success.'}
    data=json.dumps(result,indent=2)+'\n'
    if args.report=='-':print(data)
    else:pathlib.Path(args.report).write_text(data)
if __name__=='__main__':main()
