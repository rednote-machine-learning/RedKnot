"""Small deterministic natural-source qualification suite, never task results.

No source text is executed, repeated as padding or truncated to fill a target.
The duplicate occurrence is an explicit reread test. Long coverage uses whole
different Python files until it crosses the seventeen-chunk group boundary.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from kv_cache_engram.core import digest, require
from .admission import plan_request, MAX_BANK_BYTES


TOOLS = [dict(type='function', function=dict(name='read_file', description='Read repository source',
    parameters=dict(type='object', properties=dict(path=dict(type='string')), required=['path'])))]
CPU_SCOPE = dict(model_sha256='a'*64, execution_sha256='b'*64, tokenizer_sha256='c'*64)


def _source(root, path):
    raw = (root/path).read_bytes()
    return dict(path=path, text=raw.decode('utf-8'), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def _payload(question, sources, agent_payload=None):
    calls = [dict(id=f'read{i}', type='function', function=dict(name='read_file',
             arguments=json.dumps(dict(path=source['path']), separators=(',', ':')))) for i, source in enumerate(sources)]
    if agent_payload is None:
        prefix = [dict(role='system', content='You inspect repository code. Use the supplied source and explain briefly.')]
        tools = copy.deepcopy(TOOLS)
    else:
        require(type(agent_payload.get('messages')) is list and agent_payload['messages']
                and type(agent_payload.get('tools')) is list, 'actual captured agent messages/tools required')
        prefix = copy.deepcopy(agent_payload['messages'])
        tools = copy.deepcopy(agent_payload['tools'])
        # Qualification does not execute these diagnostic read actions; retain
        # the actual tool schema instead of inserting an artificial read tool.
        function = next((t['function']['name'] for t in tools if 'terminal' in t['function']['name'].lower()), None)
        require(function is not None, 'actual agent terminal tool schema required')
        for call, source in zip(calls, sources):
            call['function'] = dict(name=function, arguments=json.dumps(dict(command='cat '+source['path']), separators=(',', ':')))
    messages = prefix + [dict(role='user', content=question),
                        dict(role='assistant', content='', reasoning_content='Inspect the requested source files.', tool_calls=calls)]
    messages += [dict(role='tool', tool_call_id=f'read{i}', content=source['text']) for i, source in enumerate(sources)]
    return dict(messages=messages, tools=tools, temperature=0.0, top_p=1.0,
                max_tokens=32, stream=False)


def rendered_requests(codec, source_root=None, agent_payload=None):
    root = Path(source_root or Path(__file__).resolve().parents[1]/'NativeSnapshot')
    first = _source(root, 'deepinfer/entrypoints/llm.py')
    second = _source(root, 'deepinfer/sampling_params.py')
    short = dict(second, path=second['path']+':first-line', text=second['text'].splitlines(keepends=True)[0])
    short['sha256'] = hashlib.sha256(short['text'].encode()).hexdigest()
    short['bytes'] = len(short['text'].encode())

    def render(case_id, payload, sources, expected_route):
        value = codec.render(payload)
        plan = plan_request(value['input_ids'], value['documents'], CPU_SCOPE, bank_budget_bytes=MAX_BANK_BYTES)
        require(plan['route'] == expected_route, 'qualification source geometry did not exercise declared route: '+case_id)
        return dict(case_id=case_id, payload=payload, input_ids=list(value['input_ids']),
                    input_ids_sha256=value['input_ids_sha256'], documents=value['documents'],
                    encoder_sha256=value['encoder_sha256'], expected_route=expected_route,
                    source_refs=[{k: s[k] for k in ('path', 'sha256', 'bytes')} for s in sources],
                    expected_unique_keys=plan['unique_document_ids'],
                    actual_agent_prefix_sha256=None if agent_payload is None else digest(agent_payload),
                    diagnostic_only=True, official_task=False, gpu_executed=False)

    cold = render('natural_cold', _payload('Explain how this module submits and drains generation requests.', [first, short], agent_payload),
                  [first, short], 's2')
    repeated_sources = [second, first, first]
    repeated = render('cross_prefix_repeated', _payload('Compare sampling controls with two separate reads of the generation interface.', repeated_sources, agent_payload),
                      repeated_sources, 's2')
    cold_keys = [d['document_id'] for d in cold['documents'] if d['tokens'] >= 256 and d['origin'].get('role') == 'tool'
                 and d['origin'].get('tool_call_id') == 'read0']
    repeat_keys = [d['document_id'] for d in repeated['documents'] if d['tokens'] >= 256 and d['origin'].get('role') == 'tool']
    require(len(cold_keys) == 1 and repeat_keys.count(cold_keys[0]) == 2,
            'same natural source must retain exact token identity at both repeated occurrences')
    require(cold['input_ids_sha256'] != repeated['input_ids_sha256'], 'cross-prefix case must change the full request')
    cold_at = next(d['input_start'] for d in cold['documents'] if d['document_id'] == cold_keys[0])
    require(all(d['input_start'] != cold_at for d in repeated['documents'] if d['document_id'] == cold_keys[0]),
            'reused document must move in the rendered prompt')

    selected = []
    candidates = sorted(root.rglob('*.py'), key=lambda p: (-p.stat().st_size, str(p.relative_to(root))))
    long = None
    for path in candidates:
        source = _source(root, str(path.relative_to(root)))
        selected.append(source)
        payload = _payload('Review these complete repository modules and summarize the execution flow.', selected, agent_payload)
        candidate = codec.render(payload)
        if len(candidate['input_ids']) > 17*8192:
            require(len(candidate['input_ids']) < 270336-32, 'natural qualification corpus exceeds model capacity')
            long = render('group17_partial', payload, selected, 's2')
            break
    require(long is not None, 'insufficient distinct natural source to exercise group17 plus partial group')
    short_case = render('short_only', dict(messages=[dict(role='user', content=short['text'])],
        temperature=0.0, top_p=1.0, max_tokens=32, stream=False), [short], 'full_fallback')
    return [cold, repeated, long, short_case]


def append_actual_response(codec, earlier_case, actual_assistant_message):
    """Follow-up includes an actual generated message, not a scripted answer."""
    require(type(actual_assistant_message) is dict and actual_assistant_message.get('role') == 'assistant'
            and type(actual_assistant_message.get('content')) is str
            and not actual_assistant_message.get('tool_calls'), 'completed actual assistant message required for diagnostic follow-up')
    payload = copy.deepcopy(earlier_case['payload'])
    payload['messages'] += [copy.deepcopy(actual_assistant_message),
        dict(role='user', content='Using the same source, state one concrete constraint of that implementation.')]
    rendered = codec.render(payload)
    return dict(case_id='actual_decode_followup', payload=payload, input_ids=list(rendered['input_ids']),
                input_ids_sha256=rendered['input_ids_sha256'], documents=rendered['documents'],
                encoder_sha256=rendered['encoder_sha256'], source_refs=earlier_case['source_refs'],
                expected_route='s2', prior_case_id=earlier_case['case_id'],
                actual_assistant_message_sha256=digest(actual_assistant_message),
                diagnostic_only=True, official_task=False, gpu_executed=False)
