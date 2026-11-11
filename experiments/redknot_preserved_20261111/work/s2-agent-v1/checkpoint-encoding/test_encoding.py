"""
Tests for encoding.py (DeepSeek-V4.1 encoding).

Adapted from dsv41-master/deepseek_harmony/tests/test_deepseek_v41.py for the
self-contained dict-based API in this repo.
"""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import encoding as enc
from encoding import (
    IMAGE_PLACEHOLDER,
    SYSTEM_SP_TOKEN,
    encode_messages,
    parse_message_from_completion_text,
    render_message,
    merge_tool_messages,
)


REASONING_EFFORT_TEMPLATE = (
    SYSTEM_SP_TOKEN + "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)

V41_TOOL_CALL_OUTPUT = (
    '  reason  </think>summary\n\n'
    '<｜DSML｜ calls>\n'
    '<｜DSML｜ invoke name="lookup">\n'
    '<｜DSML｜ parameter name="query" string="true">value'
    '</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="limit" string="false">2'
    '</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n'
    '</｜DSML｜ calls><｜end▁of▁sentence｜>'
)


def make_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
        },
    }


def make_tool_call_messages() -> list:
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "reasoning_content": "  reason  ",
            "content": "summary",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"value","limit":2}',
                    },
                }
            ],
        },
    ]


# ============================================================
# Vision
# ============================================================

def test_v41_renders_images() -> None:
    prompt, media = encode_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {"type": "image_url", "image_url": {"url": "/unused/image.png"}},
                ],
            }
        ],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )

    assert prompt == (
        '<｜begin▁of▁sentence｜><｜User｜>inspect\n\n'
        f'{IMAGE_PLACEHOLDER}<｜Assistant｜></think>'
    )
    assert media == {"images": [{"type": "image", "url": "/unused/image.png"}]}


@pytest.mark.parametrize("role", ["user", "tool"])
def test_v41_rejects_image_placeholder_in_text(role: str) -> None:
    with pytest.raises(ValueError):
        encode_messages(
            [{"role": role, "content": f"hi {IMAGE_PLACEHOLDER}"}],
            thinking_mode="chat",
        )


@pytest.mark.parametrize(
    "input_format",
    ["content_blocks", "tool_messages", "tool_content_blocks", "context"],
)
def test_v41_tool_result_images_follow_prompt_order(input_format: str) -> None:
    history = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "history.png"}},
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": name,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
                for name in ("a", "b")
            ],
        },
    ]
    results = [
        {
            "type": "tool_result",
            "tool_use_id": name,
            "content": [
                {"type": "text", "text": f"result {name}"},
                {"type": "image_url", "image_url": {"url": f"{name}.png"}},
            ],
        }
        for name in ("b", "a")
    ]
    if input_format in ("tool_messages", "tool_content_blocks"):
        content_field = (
            "content_blocks" if input_format == "tool_content_blocks" else "content"
        )
        messages = history + [
            {
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                content_field: block["content"],
            }
            for block in results
        ]
        context = []
    elif input_format == "context":
        messages = [{"role": "user", "content_blocks": results}]
        context = history
    else:
        messages = history + [{"role": "user", "content_blocks": results}]
        context = []
    original = copy.deepcopy((messages, context))

    prompt, media = encode_messages(
        messages,
        thinking_mode="chat",
        context=context,
        return_multi_modal_data=True,
    )

    assert prompt.endswith(
        f"<｜User｜><tool_result>result a\n\n{IMAGE_PLACEHOLDER}</tool_result>\n\n"
        f"<tool_result>result b\n\n{IMAGE_PLACEHOLDER}</tool_result><｜Assistant｜></think>"
    )
    expected_urls = ["a.png", "b.png"] if context else ["history.png", "a.png", "b.png"]
    assert media == {"images": [{"type": "image", "url": url} for url in expected_urls]}
    assert (messages, context) == original


# ============================================================
# Reasoning Effort
# ============================================================

@pytest.mark.parametrize(
    ("effort", "budget"),
    [
        (None, 75),
        ("low", 50),
        ("high", 75),
        ("max", 100),
        (1, 1),
        (42, 42),
        (100, 100),
    ],
)
def test_v41_maps_reasoning_effort_to_1_100_budget(
    effort: Any,
    budget: int,
) -> None:
    prompt = encode_messages(
        [{"role": "user", "content": "question"}],
        thinking_mode="thinking",
        reasoning_effort=effort,
    )

    assert prompt == (
        '<｜begin▁of▁sentence｜>'
        f'{REASONING_EFFORT_TEMPLATE.format(budget=budget)}'
        '<｜User｜>question<｜Assistant｜><think>'
    )


def test_v41_only_adds_reasoning_effort_to_first_thinking_message() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]

    later_message = render_message(
        1, messages, thinking_mode="thinking", reasoning_effort=100
    )
    chat_message = render_message(
        0, messages, thinking_mode="chat", reasoning_effort=100
    )

    assert "Reasoning Effort:" not in later_message
    assert "Reasoning Effort:" not in chat_message


def test_v41_chat_mode_has_no_reasoning_effort_or_system_token() -> None:
    prompt = encode_messages(
        [{"role": "user", "content": "hello"}],
        thinking_mode="chat",
        reasoning_effort="max",
    )
    assert prompt == '<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>'


@pytest.mark.parametrize("effort", [-1, 0, 101, "medium"])
def test_v41_rejects_out_of_range_or_unknown_reasoning_effort(
    effort: Any,
) -> None:
    with pytest.raises(AssertionError, match=r"int within \[1,100\]"):
        encode_messages(
            [{"role": "user", "content": "question"}],
            thinking_mode="thinking",
            reasoning_effort=effort,
        )


@pytest.mark.parametrize("effort", [True, False, 1.5])
def test_v41_rejects_non_string_non_integer_effort_types(effort: Any) -> None:
    # bool is not `type(...) is int`; float is invalid too
    with pytest.raises(AssertionError):
        encode_messages(
            [{"role": "user", "content": "question"}],
            thinking_mode="thinking",
            reasoning_effort=effort,
        )


# ============================================================
# System token
# ============================================================

def test_v41_leading_system_message_uses_system_token() -> None:
    prompt = encode_messages(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ],
        thinking_mode="chat",
    )
    assert prompt == (
        '<｜begin▁of▁sentence｜><｜System｜>You are a helpful assistant.'
        '<｜User｜>hello<｜Assistant｜></think>'
    )


def test_v41_mid_conversation_system_message() -> None:
    prompt = encode_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "system", "content": "mid sys"},
        ],
        thinking_mode="thinking",
        reasoning_effort=88,
    )
    # Mid-conversation system gets its own <｜System｜> token and triggers
    # the assistant generation header afterwards.
    assert prompt == (
        '<｜begin▁of▁sentence｜>'
        f'{REASONING_EFFORT_TEMPLATE.format(budget=88)}'
        'sys<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>'
        '<｜System｜>mid sys<｜Assistant｜><think>'
    )


# ============================================================
# DSML tool tags
# ============================================================

def test_v41_tool_instructions_use_spaced_dsml_tags_in_chat_mode() -> None:
    prompt = encode_messages(
        [
            {"role": "system", "content": "system", "tools": [make_tool()]},
            {"role": "user", "content": "question"},
        ],
        thinking_mode="chat",
    )

    assert (
        '<｜DSML｜ calls>\n'
        '<｜DSML｜ invoke name="$TOOL_NAME">\n'
        '<｜DSML｜ parameter name="$PARAMETER_NAME" '
        'string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>\n'
        '...\n'
        '</｜DSML｜ invoke>'
    ) in prompt
    assert '<｜DSML｜tool_calls>' not in prompt
    assert '<｜DSML｜invoke' not in prompt
    assert '<｜DSML｜parameter' not in prompt


def test_v41_renders_spaced_dsml_with_v4_assistant_semantics() -> None:
    messages = make_tool_call_messages()

    prompt = render_message(1, messages, thinking_mode="thinking")

    assert prompt == V41_TOOL_CALL_OUTPUT


def test_v41_parses_spaced_dsml_roundtrip() -> None:
    messages = make_tool_call_messages()

    parsed = parse_message_from_completion_text(
        V41_TOOL_CALL_OUTPUT, thinking_mode="thinking"
    )

    assert parsed["role"] == "assistant"
    assert parsed["reasoning_content"] == "  reason  "
    assert parsed["content"] == "summary"
    assert parsed["tool_calls"]
    assert parsed["tool_calls"][0]["function"]["name"] == "lookup"
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "query": "value",
        "limit": 2,
    }

    # Re-encoding the parsed message reproduces the original completion text
    assert encode_messages(
        [parsed],
        thinking_mode="thinking",
        context=messages[:1],
    ) == V41_TOOL_CALL_OUTPUT


def test_v41_parse_rejects_unspaced_v4_dsml() -> None:
    v4_output = V41_TOOL_CALL_OUTPUT.replace("｜DSML｜ calls", "｜DSML｜tool_calls") \
        .replace("｜DSML｜ invoke", "｜DSML｜invoke") \
        .replace("｜DSML｜ parameter", "｜DSML｜parameter")
    with pytest.raises(AssertionError):
        parse_message_from_completion_text(v4_output, thinking_mode="thinking")


# ============================================================
# Tool namespaces
# ============================================================

@pytest.mark.parametrize("location", ["tool", "function"])
@pytest.mark.parametrize("namespace", ["search", {"name": "search", "description": "Search tools."}])
def test_v41_renders_namespaced_tool_schemas(location: str, namespace: Any) -> None:
    tool = make_tool()
    target = tool if location == "tool" else tool["function"]
    target["namespace"] = namespace
    original = copy.deepcopy(tool)

    prompt = encode_messages(
        [{"role": "system", "content": "system", "tools": [tool]}],
        thinking_mode="chat",
    )

    schema = dict(make_tool()["function"], name="search::lookup")
    if isinstance(namespace, dict):
        schema["description"] = "Search tools.\nLook up a value"
    assert json.dumps(schema) in prompt
    assert '"namespace":' not in prompt
    assert tool == original


@pytest.mark.parametrize("thinking_mode", ["chat", "thinking"])
@pytest.mark.parametrize("location", ["tool", "function", "qualified_name"])
def test_v41_namespaced_tool_calls_roundtrip(thinking_mode: str, location: str) -> None:
    messages = make_tool_call_messages()
    call = messages[1]["tool_calls"][0]
    if location == "qualified_name":
        call["function"]["name"] = "search::lookup"
    else:
        target = call if location == "tool" else call["function"]
        target["namespace"] = "search"
    original = copy.deepcopy(messages)

    expected = V41_TOOL_CALL_OUTPUT.replace('name="lookup"', 'name="search::lookup"')
    if thinking_mode == "chat":
        expected = expected.split("</think>", 1)[1]
    assert render_message(1, messages, thinking_mode=thinking_mode) == expected

    parsed = parse_message_from_completion_text(expected, thinking_mode=thinking_mode)
    assert parsed["tool_calls"] == [{
        "type": "function",
        "namespace": "search",
        "function": {
            "name": "lookup",
            "arguments": '{"query": "value", "limit": 2}',
        },
    }]
    assert encode_messages(
        [parsed], thinking_mode=thinking_mode, context=messages[:1]
    ) == expected
    assert messages == original


def test_v41_keeps_same_named_tools_in_separate_namespaces() -> None:
    tools, calls = [], []
    for namespace in (None, "search", "files"):
        tool = make_tool()
        call = {
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"value"}'},
        }
        if namespace is not None:
            tool["namespace"] = {"name": namespace}
            call["namespace"] = namespace
        tools.append(tool)
        calls.append(call)

    messages = [
        {"role": "system", "content": "system", "tools": tools},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "summary", "tool_calls": calls},
    ]
    prompt = encode_messages(messages, thinking_mode="chat")
    for name in ("lookup", "search::lookup", "files::lookup"):
        assert f'"name": "{name}"' in prompt
        assert f'<｜DSML｜ invoke name="{name}">' in prompt

    completion = render_message(2, messages, thinking_mode="chat")
    parsed = parse_message_from_completion_text(completion, thinking_mode="chat")
    assert "namespace" not in parsed["tool_calls"][0]
    assert [call.get("namespace") for call in parsed["tool_calls"]] == [None, "search", "files"]
    assert all(call["function"]["name"] == "lookup" for call in parsed["tool_calls"])


def test_v41_does_not_duplicate_a_qualified_namespace() -> None:
    tool = make_tool()
    tool["function"]["name"] = "search::lookup"
    tool["namespace"] = {"name": "search", "description": "Search tools."}
    schema = enc.tools_from_openai_format([tool])[0]
    assert schema["name"] == "search::lookup"
    assert schema["description"] == "Search tools.\nLook up a value"

    messages = make_tool_call_messages()
    call = messages[1]["tool_calls"][0]
    call["function"]["name"] = "search::lookup"
    call["namespace"] = "search"
    assert render_message(1, messages, thinking_mode="thinking") == (
        V41_TOOL_CALL_OUTPUT.replace('name="lookup"', 'name="search::lookup"')
    )


@pytest.mark.parametrize(
    ("name", "namespace", "error"),
    [
        ("search::lookup", "files", "Conflicting tool namespaces"),
        ("search::nested::lookup", None, "Tool name must not contain"),
        ("lookup", "search::nested", "Tool namespace must not contain"),
    ],
)
def test_v41_rejects_ambiguous_tool_namespaces(name: str, namespace: Any, error: str) -> None:
    tool = make_tool()
    tool["function"]["name"] = name
    tool["namespace"] = namespace
    with pytest.raises(AssertionError, match=error):
        enc.tools_from_openai_format([tool])

    messages = make_tool_call_messages()
    call = messages[1]["tool_calls"][0]
    call["function"]["name"] = name
    call["namespace"] = namespace
    with pytest.raises(AssertionError, match=error):
        render_message(1, messages, thinking_mode="thinking")


# ============================================================
# Multi-turn flow
# ============================================================

def test_v41_drop_thinking_without_tools() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "reasoning_content": "r2",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    prompt = encode_messages(
        messages,
        thinking_mode="thinking",
        drop_thinking=True,
    )
    # Drop completed-turn reasoning, but keep the current turn across tool results.
    assert "<｜User｜>q1<｜Assistant｜></think>a1<｜end▁of▁sentence｜>" in prompt
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>")

    next_turn = encode_messages(
        messages + [{"role": "user", "content": "q3"}],
        thinking_mode="thinking",
    )
    assert "r2" not in next_turn
    assert next_turn.endswith("q3<｜Assistant｜><think>")


# ============================================================
# Preprocessing
# ============================================================

def test_merge_tool_messages_creates_tool_result_blocks() -> None:
    merged = merge_tool_messages([
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "a", "content": "r1"},
        {"role": "tool", "tool_call_id": "b", "content": "r2"},
    ])
    assert len(merged) == 2
    assert merged[1]["role"] == "user"
    assert [b["type"] for b in merged[1]["content_blocks"]] == ["tool_result", "tool_result"]


def test_v41_task_sp_token() -> None:
    prompt = encode_messages(
        [{"role": "user", "content": "classify me", "task": "query"}],
        thinking_mode="chat",
    )
    assert prompt.endswith("classify me<｜query｜>")
    assert "<｜Assistant｜>" not in prompt


# ============================================================
# Golden fixtures from encoding/tests
# ============================================================

ENCODING_DIR = Path(__file__).resolve().parent
ENCODING_FIXTURES_DIR = ENCODING_DIR / "tests"
INFERENCE_EXAMPLES_DIR = ENCODING_DIR.parent / "inference" / "examples"

FIXTURE_CASE_IDS = sorted(
    int(p.stem.split("_")[-1])
    for p in ENCODING_FIXTURES_DIR.glob("test_input_*.json")
)


@pytest.mark.parametrize("case_id", FIXTURE_CASE_IDS)
def test_examples_encoding_golden_outputs(case_id: int) -> None:
    """Each tests/encoding input must encode to its checked-in golden output."""
    input_file = ENCODING_FIXTURES_DIR / f"test_input_{case_id}.json"
    output_file = ENCODING_FIXTURES_DIR / f"test_output_{case_id}.txt"
    assert output_file.exists(), f"missing golden output: {output_file.name} (run tests/encoding/regen_outputs.py)"

    case = enc.load_cases(str(input_file))[0]
    prompt, _ = enc.encode_case(case, thinking_mode="chat")

    assert prompt == output_file.read_text(), (
        f"{output_file.name} is stale; regenerate with tests/encoding/regen_outputs.py"
    )


def test_examples_v41_output_uses_v41_format_markers() -> None:
    """Sanity-check the V4.1 goldens actually exercise V4.1-specific format."""
    # case 1: tool calls with spaced DSML tags
    out1 = (ENCODING_FIXTURES_DIR / "test_output_1.txt").read_text()
    assert '<｜DSML｜ calls>' in out1 and '<｜DSML｜ invoke name="get_weather">' in out1
    assert '<｜DSML｜tool_calls>' not in out1

    # case 5: numeric reasoning effort behind the system token
    out5 = (ENCODING_FIXTURES_DIR / "test_output_5.txt").read_text()
    assert out5.startswith(
        '<｜begin▁of▁sentence｜>' + REASONING_EFFORT_TEMPLATE.format(budget=100)
    )
    assert out5.count(IMAGE_PLACEHOLDER) == 2


def test_examples_vl_txt_and_json_encode_identically() -> None:
    """The TXT (last block of example.txt) and JSON vision examples must encode identically."""
    txt = (INFERENCE_EXAMPLES_DIR / "example.txt").read_text().rstrip("\n").split("\n\n")[-1]
    messages = [{"role": "user", "content": enc.parse_tagged_text(txt)}]
    p1, m1 = encode_messages(messages, thinking_mode="chat", return_multi_modal_data=True)

    case = enc.load_cases(str(INFERENCE_EXAMPLES_DIR / "example_harmony.json"))[0]
    p2, m2 = enc.encode_case(case, thinking_mode="chat")

    assert p1 == p2
    assert m1["images"] == m2
    assert len(m2) == 2


def test_examples_harmony_cases_encode() -> None:
    """All example_harmony.json cases encode without error."""
    cases = enc.load_cases(str(INFERENCE_EXAMPLES_DIR / "example_harmony.json"))
    assert len(cases) == 4

    # case 1 (vision) is covered by test_examples_vl_txt_and_json_encode_identically

    # cases are pure OpenAI format: mode/effort are passed at call time
    prompt = encode_messages(
        cases[1]["messages"], thinking_mode="thinking", reasoning_effort=75
    )
    assert REASONING_EFFORT_TEMPLATE.format(budget=75) in prompt

    # case 3: tools with spaced DSML tags
    prompt, _ = enc.encode_case(cases[2], thinking_mode="chat")
    assert '<｜DSML｜ calls>' in prompt

    # case 4: mid-conversation system message triggers assistant header
    prompt, _ = enc.encode_case(cases[3], thinking_mode="chat")
    assert '<｜System｜>Mid-conversation instruction update' in prompt
    assert prompt.endswith('<｜Assistant｜></think>')


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
