import ast
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from uuid import UUID

B=Path(__file__).parent
sys.path.insert(0,str(B/'runtime_guard'))
import routing_guard as guard


class DispatchGuardTests(unittest.TestCase):
    def test_complete_pinned_chat_options_and_original_prepare_preserve_official_defaults(self):
        sdk=B/'benchmarks/vendor/software-agent-sdk/openhands-sdk/openhands/sdk/llm'
        source=sdk/'llm.py';tree=ast.parse(source.read_text())
        functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('_openrouter_headers','_prepare_transport_kwargs')]
        defaults={n.target.id:ast.literal_eval(next(k.value for k in n.value.keywords if k.arg=='default')) for n in ast.walk(tree) if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in ('openrouter_site_url','openrouter_app_name')}
        common=ast.parse((sdk/'options/common.py').read_text())
        chat=ast.parse((sdk/'options/chat_options.py').read_text())
        functions += [n for n in common.body+chat.body if isinstance(n,ast.FunctionDef) and n.name in ('apply_defaults_if_absent','select_chat_options')]
        ns={'Any':object,'get_features':lambda model:types.SimpleNamespace(supports_reasoning_effort=False,supports_extended_thinking=False,supports_prompt_cache_retention=False)}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),ns)
        config=json.loads((B/'config/llm.Full.json').read_text())
        sid='6d7ef556-bf6c-47f0-ae73-9cb569b60c78'
        llm=types.SimpleNamespace(**{**config,**defaults,'extra_headers':{'x-litellm-session-id':sid},'top_k':None,'effective_max_output_tokens':None,'_prompt_cache_key':sid,'api_version':None,'litellm_extra_body':{}})
        llm._openrouter_headers=types.MethodType(ns['_openrouter_headers'],llm)
        llm._litellm_call_kwargs=lambda:dict(model=llm.model,api_base=llm.base_url)
        llm._get_litellm_api_key_value=lambda:'local-cpu-fixture'
        llm._aws_kwargs=lambda:{}
        tools=[{'type':'function','function':{'name':'finish'}}]
        options=ns['select_chat_options'](llm,{'tools':tools},True)
        final=ns['_prepare_transport_kwargs'](llm,messages=[],enable_streaming=False,**options)
        guard.validate_llm(llm);self.assertIs(guard.validate_transport(final),final)
        self.assertEqual(final['extra_headers'],{'HTTP-Referer':defaults['openrouter_site_url'],'X-Title':defaults['openrouter_app_name'],'x-litellm-session-id':sid})
        self.assertEqual(final['tools'],tools);self.assertEqual(final['prompt_cache_key'],sid)
        for key in ('max_tokens','max_completion_tokens','max_output_tokens'):self.assertNotIn(key,final)
        # Additional harmless observability headers must not trigger another
        # metadata-name allowlist failure. Routing and auth remain rejected.
        final['extra_headers'].update({'traceparent':'00-'+32*'a'+'-'+16*'b'+'-01','X-Request-ID':'cpu-observation'})
        self.assertIs(guard.validate_transport(final),final)
        for field,value in (('model','openai/other'),('api_base','https://external.invalid'),('temperature',1),('top_p',0.5),('seed',99)):
            with self.assertRaises(ValueError):guard.validate_transport({**final,field:value})

    def test_actual_conversation_metadata_is_preserved_and_not_a_policy_override(self):
        source=B/'benchmarks/vendor/software-agent-sdk/openhands-sdk/openhands/sdk/conversation/impl/local_conversation.py'
        tree=ast.parse(source.read_text())
        methods=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('_pin_prompt_cache_key','_pin_session_affinity_header')]
        self.assertEqual(len(methods),2)
        ns={'LLM':types.SimpleNamespace}
        exec(compile(ast.Module(body=methods,type_ignores=[]),str(source),'exec'),ns)
        config=json.loads((B/'config/llm.Full.json').read_text())
        llm=types.SimpleNamespace(**{**config,'extra_headers':None},_prompt_cache_key=None)
        sid=UUID('6d7ef556-bf6c-47f0-ae73-9cb569b60c78')
        owner=types.SimpleNamespace(agent=types.SimpleNamespace(llm=llm),_state=types.SimpleNamespace(id=sid))
        ns['_pin_prompt_cache_key'](owner);ns['_pin_session_affinity_header'](owner,llm)
        guard.validate_llm(llm)
        self.assertEqual(llm.extra_headers,{'x-litellm-session-id':str(sid)})
        self.assertEqual(llm._prompt_cache_key,str(sid))
        kwargs=dict(model=guard.MODEL,api_base=guard.BASE_URL,temperature=0,top_p=1,seed=0,extra_headers=llm.extra_headers,prompt_cache_key=llm._prompt_cache_key)
        self.assertIs(guard.validate_transport(kwargs),kwargs)
        self.assertIs(kwargs['extra_headers'],llm.extra_headers)
        self.assertEqual(kwargs['prompt_cache_key'],str(sid))
        for headers in ({'Authorization':'Bearer wrong'},{'Host':'other.invalid'},{'X-Forwarded-Host':'other.invalid'},{'x-litellm-session-id':'bad\r\nHost:other'}, {'x-litellm-session-id':str(sid),'X-LiteLLM-Session-ID':str(sid)}):
            with self.assertRaises(ValueError):guard.validate_transport({**kwargs,'extra_headers':headers})
            llm.extra_headers=headers
            with self.assertRaises(ValueError):guard.validate_llm(llm)

    def fake_sdk(self):
        class LLM:
            prepare_calls=0
            def __init__(self,**kwargs):
                self.__dict__.update(kwargs)
                self.effective_max_output_tokens=None
                self._init_model_info_and_caps()
            def _init_model_info_and_caps(self):
                self.original_constructor_ran=True
            def _prepare_transport_kwargs(self,**kwargs):
                type(self).prepare_calls+=1
                return dict(model=self.model,api_base=self.base_url,temperature=self.temperature,top_p=self.top_p,seed=self.seed,**kwargs)
            def _build_responses_call_kwargs(self,*args,**kwargs):
                raise AssertionError('Original Responses transport reached')
        modules={name:types.ModuleType(name) for name in ('openhands','openhands.sdk','openhands.sdk.llm')}
        modules['openhands.sdk.llm'].LLM=LLM
        return LLM,modules

    def test_actual_official_documentation_constructor_is_inert_but_dispatch_blocked(self):
        LLM,modules=self.fake_sdk();original_init=LLM._init_model_info_and_caps
        source=B/'benchmarks/vendor/software-agent-sdk/openhands-agent-server/openhands/agent_server/conversation_router.py'
        tree=ast.parse(source.read_text())
        constructors=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='LLM']
        self.assertEqual(len(constructors),1)
        with patch.dict(sys.modules,modules):
            self.assertIs(guard.install(),LLM)
            self.assertIs(LLM._init_model_info_and_caps,original_init)
            example=eval(compile(ast.Expression(constructors[0]),str(source),'eval'),{'LLM':LLM,'SecretStr':lambda s:s})
            self.assertEqual(example.model,'your-model-provider/your-model-name')
            self.assertTrue(example.original_constructor_ran)
            with self.assertRaisesRegex(ValueError,'S2 LLM policy mismatch: model'):
                example._prepare_transport_kwargs()
            self.assertEqual(LLM.prepare_calls,0)

    def test_valid_requests_pass_but_inferred_caps_and_responses_fail_before_transport(self):
        LLM,modules=self.fake_sdk()
        config=json.loads((B/'config/llm.Full.json').read_text())
        with patch.dict(sys.modules,modules):
            guard.install();guard.install()
            llm=LLM(**config)
            result=llm._prepare_transport_kwargs(messages=[])
            self.assertEqual(result['api_base'],guard.BASE_URL)
            self.assertEqual(LLM.prepare_calls,1)
            llm.effective_max_output_tokens=32
            with self.assertRaisesRegex(ValueError,'SDK inferred an artificial output cap'):
                llm._prepare_transport_kwargs(messages=[])
            self.assertEqual(LLM.prepare_calls,1)
            with self.assertRaisesRegex(ValueError,'only the local chat-completions'):
                llm._build_responses_call_kwargs()


if __name__=='__main__':unittest.main()
