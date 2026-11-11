"""Pinned checkpoint-owned V4.1 chat/tool codec; no substitute chat template."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import FunctionType

try:
    from .engine_bridge import token_digest
except ImportError:
    from engine_bridge import token_digest

ENCODER_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"


def load_encoder(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != ENCODER_SHA256:
        raise ValueError("checkpoint chat encoder source changed")
    spec = importlib.util.spec_from_file_location("s2_checkpoint_encoding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def token_encoding(tokenizer, text):
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    encoded = backend.encode(text, add_special_tokens=False)
    if not hasattr(encoded, "ids") or not hasattr(encoded, "offsets"):
        raise TypeError("fast tokenizer offsets required for explicit document spans")
    return tuple(encoded.ids), tuple(encoded.offsets)


def decode_ids(tokenizer, ids):
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    return backend.decode(list(ids), skip_special_tokens=False)


class CheckpointCodec:
    def __init__(self, encoder_path, tokenizer, *, thinking_mode="thinking", reasoning_effort=75):
        if thinking_mode not in ("thinking", "chat"):
            raise ValueError("unsupported thinking mode")
        self.encoder = load_encoder(encoder_path)
        self.tokenizer, self.thinking_mode, self.reasoning_effort = tokenizer, thinking_mode, reasoning_effort

    def _render(self, messages, *, encoder=None):
        encode = self.encoder.encode_messages if encoder is None else encoder
        text, media = encode(messages, thinking_mode=self.thinking_mode,
            reasoning_effort=self.reasoning_effort, return_multi_modal_data=True)
        if media.get("images"):
            raise ValueError("S2 code-agent service is text-only")
        return text

    def render(self, payload):
        effort = payload.get("reasoning_effort")
        normalized_effort = {"low": 50, "high": 75, "max": 100}.get(effort, effort)
        if effort is not None and normalized_effort != self.reasoning_effort:
            raise ValueError("request reasoning effort differs from frozen service configuration")
        if payload.get("response_format") not in (None, {"type": "text"}):
            raise ValueError("structured response formatting is not silently ignored")
        messages = copy.deepcopy(payload.get("messages"))
        if not isinstance(messages, list) or not messages:
            raise ValueError("nonempty messages required")
        for message in messages:
            if message.get("role") not in ("system", "user", "assistant", "tool"):
                raise ValueError("unsupported message role")
            content = message.get("content")
            if content is not None and not isinstance(content, (str, list)):
                raise ValueError("unsupported message content")
            if isinstance(content, list):
                if any(not isinstance(b, dict) or b.get("type") != "text"
                       or not isinstance(b.get("text"), str) for b in content):
                    raise ValueError("only text content blocks supported")
                if message["role"] in ("system", "assistant"):
                    message["content"] = "\n\n".join(b["text"] for b in content)
        tools = payload.get("tools") or []
        if any(t.get("type") != "function" or not isinstance(t.get("function"), dict) for t in tools):
            raise ValueError("only function tools supported")
        if tools:
            if messages[0]["role"] != "system":
                messages.insert(0, dict(role="system", content=""))
            messages[0]["tools"] = copy.deepcopy(tools)
        if payload.get("tool_choice") not in (None, "auto"):
            raise ValueError("tool_choice forcing is not supported by checkpoint codec")
        if payload.get("stop") not in (None, [], ""):
            raise ValueError("custom stop strings are not silently ignored")
        prompt = self._render(messages)
        ids, offsets = token_encoding(self.tokenizer, prompt)
        if not ids or ids[0] != 0:
            raise ValueError("checkpoint prompt must have exactly its own BOS")

        # Source-to-render mapping is proven by a second marker pass. No search
        # for a guessed occurrence of the body; duplicate bodies stay distinct.
        marked = copy.deepcopy(messages)
        sources = []
        nonce = hashlib.sha256(prompt.encode()).hexdigest()
        for mi, message in enumerate(marked):
            if message["role"] in ("system", "user", "tool") and isinstance(message.get("content"), str):
                origin = dict(message_index=mi, role=message["role"],
                    source_kind={"system":"system_body", "user":"user_body", "tool":"tool_result"}[message["role"]])
                if message["role"] == "tool":
                    origin["tool_call_id"] = message.get("tool_call_id")
                sources.append((message, "content", origin))
            elif message["role"] in ("user", "tool") and isinstance(message.get("content"), list):
                for bi, block in enumerate(message["content"]):
                    origin = dict(message_index=mi, content_block_index=bi, role=message["role"],
                        source_kind=message["role"]+"_text_block")
                    if message["role"] == "tool":
                        origin["tool_call_id"] = message.get("tool_call_id")
                    sources.append((block, "text", origin))
        markers = {}
        def mark(body, origin):
            ordinal = len(markers)
            start, end = f"S2START{nonce}X{ordinal}Z", f"S2END{nonce}X{ordinal}Z"
            if start in prompt or end in prompt:
                raise ValueError("source marker collision")
            markers[start] = (end, origin, body)
            return start + body + end
        for obj, key, origin in sources:
            obj[key] = mark(obj[key], origin)
        # The provider appends a generated tool instruction/schema block outside
        # system content. Map that explicit renderer return value in a private
        # CPU-only function namespace; never patch the original encoder module.
        namespace = dict(vars(self.encoder))
        for name, fn in vars(self.encoder).items():
            if isinstance(fn, FunctionType) and fn.__module__ == self.encoder.__name__:
                copied = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
                copied.__kwdefaults__ = fn.__kwdefaults__
                namespace[name] = copied
        original_tools = self.encoder.render_tools
        def mapped_tools(values):
            body = original_tools(values)
            schema_sha = hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False,
                separators=(",", ":")).encode()).hexdigest()
            return mark(body, dict(role="system", source_kind="checkpoint_tool_template",
                tool_schema_sha256=schema_sha, tool_function_count=len(values)))
        namespace["render_tools"] = mapped_tools
        mapped = self._render(marked, encoder=namespace["encode_messages"])
        pieces, spans, cursor, output_chars = [], [], 0, 0
        while markers:
            found = [(mapped.find(start, cursor), start) for start in markers]
            if any(pos < 0 for pos, _ in found):
                raise ValueError("encoder dropped a source marker")
            pos, start = min(found)
            end, origin, expected_body = markers.pop(start)
            end_pos = mapped.find(end, pos + len(start))
            if end_pos < 0:
                raise ValueError("encoder dropped a source end marker")
            prefix, body = mapped[cursor:pos], mapped[pos+len(start):end_pos]
            if body != expected_body:
                raise ValueError("encoder transformed a cacheable source body")
            pieces += [prefix, body]
            begin = output_chars + len(prefix)
            spans.append((begin, begin+len(body), origin))
            output_chars = begin + len(body)
            cursor = end_pos + len(end)
        pieces.append(mapped[cursor:])
        if "".join(pieces) != prompt:
            raise ValueError("marked source reconstruction differs from actual prompt")
        documents = []
        for start, end, origin in spans:
            # Boundary-crossing BPE tokens stay mandatory online, never added to
            # independent content cache. The actual request IDs are unchanged.
            covered = [i for i, (a, b) in enumerate(offsets) if start <= a < b <= end]
            if not covered:
                continue
            lo, hi = covered[0], covered[-1]+1
            if covered != list(range(lo, hi)):
                raise ValueError("noncontiguous token span")
            key = token_digest(ids[lo:hi])
            documents.append(dict(input_start=lo, input_end=hi, tokens=hi-lo,
                token_ids_sha256=key, document_id=key, offline_position=0,
                online_position=lo, origin=dict(**origin, character_start=start,
                    character_end=end, source_mapping="marker_roundtrip_plus_full_prompt_token_offsets")))
        return dict(prompt=prompt, input_ids=ids, documents=documents, messages=messages,
            tools=tools, encoder_sha256=ENCODER_SHA256, input_ids_sha256=token_digest(ids),
            cache_span_policy="natural_system_user_tool_bodies_and_checkpoint_rendered_tool_template_v1",
            original_encoder_unmodified=True, actual_input_ids_unchanged=True)

    def completion_stream(self, request_id):
        return CompletionStream(self, request_id)


def safe_prefix(text, delimiters):
    """Hold incomplete special-token spelling and unfinished UTF-8 decoding."""
    text = text.rstrip("\ufffd")
    hold = 0
    for delimiter in delimiters:
        for size in range(1, min(len(delimiter), len(text))+1):
            if text.endswith(delimiter[:size]):
                hold = max(hold, size)
    return text[:-hold] if hold else text


class CompletionStream:
    def __init__(self, codec, request_id):
        self.codec, self.request_id = codec, request_id
        self.ids, self.emitted = [], {"reasoning_content": "", "content": ""}
        self.closed = False

    def feed(self, token_id, *, finished=False, finish_reason=""):
        if self.closed:
            raise ValueError("completion already closed")
        self.ids.append(token_id)
        text = decode_ids(self.codec.tokenizer, self.ids)
        encoder = self.codec.encoder
        call_start = "\n\n<｜DSML｜ calls"
        fields = {"reasoning_content": "", "content": ""}
        content = text
        if self.codec.thinking_mode == "thinking":
            if "</think>" in text:
                fields["reasoning_content"], content = text.split("</think>", 1)
            else:
                fields["reasoning_content"] = safe_prefix(text, ["</think>"])
                content = ""
        stops = [position for marker in (call_start, encoder.eos_token)
                 if (position := content.find(marker)) >= 0]
        fields["content"] = content[:min(stops)] if stops else safe_prefix(content, [call_start, encoder.eos_token])
        parsed = None
        if finished and finish_reason == "stop":
            # The checkpoint parser, not our incremental filter, owns final
            # DSML/JSON validation and complete assistant-message semantics.
            parsed = encoder.parse_message_from_completion_text(text, self.codec.thinking_mode)
            fields = {key: parsed.get(key, "") for key in self.emitted}
        elif finished and finish_reason != "length":
            raise ValueError("unsuccessful native completion")
        deltas = []
        for key, value in fields.items():
            if not value.startswith(self.emitted[key]):
                raise ValueError("decoder changed already streamed text")
            delta = value[len(self.emitted[key]):]
            if delta:
                deltas.append({key: delta})
            self.emitted[key] = value
        if parsed:
            calls = parsed.get("tool_calls", [])
            for index, call in enumerate(calls):
                call = copy.deepcopy(call)
                call.update(index=index, id=f"call_{self.request_id}_{index}")
                # Invalid JSON must fail before the agent can execute the call.
                json.loads(call["function"]["arguments"])
                deltas.append(dict(tool_calls=[call]))
            finish_reason = "tool_calls" if calls else finish_reason
        if finished:
            self.closed = True
            deltas.append(dict(finish_reason=finish_reason))
        return deltas
