"""S2 local-only SDK boundary. No request/model-content rewriting is performed."""
from __future__ import annotations
import re
MODEL = 'openai/deepseek-v4.1-flash'
WIRE_MODEL = 'deepseek-v4.1-flash'
BASE_URL = 'http://127.0.0.1:8123/v1'
CONTEXT = 270336

def validate_session_metadata(headers):
    if headers is None:return
    if not isinstance(headers,dict):raise ValueError('S2 extra_headers must be a mapping')
    # SDK/library observation and attribution headers do not choose the model,
    # endpoint or sampler. Do not impose an informational-header allowlist.
    # Keep explicit authentication, authority and HTTP framing overrides out.
    forbidden={'authorization','proxy-authorization','host','forwarded',
               'connection','proxy-connection','content-length','transfer-encoding',
               'api-key','x-api-key','openai-organization','openai-project'}
    seen=set()
    for key,value in headers.items():
        normalized=key.lower() if isinstance(key,str) else None
        if (not isinstance(key,str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+",key)
                or normalized in forbidden or normalized.startswith('x-forwarded-')
                or normalized in seen):
            raise ValueError('S2 unadmitted request header: '+str(key))
        seen.add(normalized)
        if not isinstance(value,str) or any(c in value for c in ('\r','\n','\0')):
            raise ValueError('S2 request header must be single-line text')

def validate_llm(llm):
    expected = {'model': MODEL, 'base_url': BASE_URL, 'temperature': 0,
                'top_p': 1, 'seed': 0, 'max_input_tokens': CONTEXT,
                'max_output_tokens': None, 'reasoning_effort': None,
                'native_tool_calling': True, 'stream': False}
    for key, value in expected.items():
        if getattr(llm, key, None) != value:
            raise ValueError(f'S2 LLM policy mismatch: {key}')
    for key in ('fallback_strategy', 'litellm_extra_body',
                'model_canonical_name', 'custom_tokenizer'):
        if getattr(llm, key, None):
            raise ValueError(f'S2 unadmitted LLM override: {key}')
    # Preserve official affinity/attribution/observation metadata verbatim.
    validate_session_metadata(getattr(llm,'extra_headers',None))

def validate_transport(kwargs):
    expected = {'model': MODEL, 'api_base': BASE_URL, 'temperature': 0,
                'top_p': 1, 'seed': 0}
    for key, value in expected.items():
        if kwargs.get(key) != value:
            raise ValueError(f'S2 transport policy mismatch: {key}')
    for key in ('max_tokens', 'max_completion_tokens', 'max_output_tokens',
                'stop', 'reasoning_effort', 'extra_body', 'fallbacks',
                'custom_llm_provider', 'api_version'):
        if kwargs.get(key) is not None:
            raise ValueError(f'S2 unadmitted transport argument: {key}')
    if kwargs.get('tool_choice') not in (None, 'auto'):
        raise ValueError('S2 supports only automatic tool choice')
    validate_session_metadata(kwargs.get('extra_headers'))
    return kwargs

def install():
    from openhands.sdk.llm import LLM
    if getattr(LLM, '_s2_local_route_guard', False):
        return LLM
    original_prepare = LLM._prepare_transport_kwargs
    def prepare(self, *args, **kwargs):
        # Official router imports construct inert documentation examples. Policy
        # applies at the actual transport boundary, without a placeholder bypass.
        validate_llm(self)
        if self.effective_max_output_tokens is not None:
            raise ValueError('SDK inferred an artificial output cap; admission required')
        return validate_transport(original_prepare(self, *args, **kwargs))
    def responses_forbidden(self, *args, **kwargs):
        raise ValueError('S2 admits only the local chat-completions route')
    LLM._prepare_transport_kwargs = prepare
    LLM._build_responses_call_kwargs = responses_forbidden
    LLM._s2_local_route_guard = True
    return LLM
