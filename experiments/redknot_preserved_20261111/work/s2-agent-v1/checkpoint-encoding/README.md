# DeepSeek-V4.1 text and vision encoding

`encoding.py` is the standalone prompt-format reference for DeepSeek-V4.1. It
supports multi-turn conversations, tool calls, thinking modes, numeric reasoning
effort, mid-conversation system messages, and interleaved image content blocks,
without importing the inference implementation.

## V4.1 changes relative to V4

Three prompt-format changes distinguish V4.1 from V4:

1. **DSML tag names use a leading space.** Tool calls are wrapped in
   `<｜DSML｜ calls>` blocks with `<｜DSML｜ invoke>` / `<｜DSML｜ parameter>` tags
   (note the space before `calls`, `invoke`, and `parameter`). The V4 format used
   `<｜DSML｜tool_calls>` without a space.

2. **Reasoning effort is a numeric budget (1–100).** The effort prefix is
   rendered as `Reasoning Effort: {budget} (range 1-100, ...)` rather than the
   verbose natural-language descriptions used in V4. String aliases map as
   follows: `"low"` → 50, `"high"` → 75, `"max"` → 100. The
   default is `"high"` (75). The effort prefix is only rendered in
   `thinking_mode="thinking"` and only at the beginning of the conversation
   (index 0).

3. **Mid-conversation system messages** are supported via the `<｜System｜>` token.
   A mid-conversation system message behaves like a user message for the purpose
   of appending the assistant generation header.

## Quick start

```python
from encoding import encode_messages, parse_message_from_completion_text

# Text-only conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
]
prompt, media = encode_messages(
    messages,
    thinking_mode="thinking",
    reasoning_effort=75,           # integer 1–100, or "low"/"high"/"max"
    return_multi_modal_data=True,
)
# prompt:
# '<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the
#  value, the more thorough the reasoning)\n\nYou are a helpful assistant.
#  <｜User｜>What is 2+2?<｜Assistant｜><think>'

# Parse model output back to a structured message
completion = "Simple arithmetic.</think>2 + 2 = 4.<｜end▁of▁sentence｜>"
parsed = parse_message_from_completion_text(completion, thinking_mode="thinking")
# => {"role": "assistant", "reasoning_content": "Simple arithmetic.",
#     "content": "2 + 2 = 4.", "tool_calls": []}
```

> **Note:** `parse_message_from_completion_text` is designed to handle
> well-formatted model output only. It does not attempt to correct or recover
> from malformed output that the model might occasionally generate. For
> production use, additional error handling is recommended.

## OpenAI-style messages

```python
from encoding import encode_messages

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "第一张图"},
        {
            "type": "image_url",
            "image_url": {"url": "examples/images/image_1.jpeg"},
        },
        {"type": "text", "text": "有什么内容？"},
    ],
}]

prompt, media = encode_messages(
    messages,
    thinking_mode="chat",
    return_multi_modal_data=True,
)
# prompt:
# '<｜begin▁of▁sentence｜><｜User｜>第一张图\n\n<｜deepseek_image｜>\n\n有什么内容？<｜Assistant｜></think>'
# media["images"] contains the image records in prompt order
```

Images are represented in the prompt by `<｜deepseek_image｜>`. `media["images"]`
contains the corresponding image records in exactly the same order they appear in
the prompt. Pixel loading and expansion into model image tokens are handled by
`inference/image_processor.py`.

## Compact TXT notation

`parse_tagged_text()` converts a compact prompt such as

```text
第一张图<image>examples/images/image_1.jpeg</image>有什么内容？
```

into the same standard content blocks. It is an input convenience layer, not a
second encoding implementation.

## Message format

### Special tokens

| Token | Purpose |
| :--- | :--- |
| `<｜begin▁of▁sentence｜>` | Beginning of sequence (BOS) |
| `<｜end▁of▁sentence｜>` | End of assistant turn (EOS) |
| `<｜User｜>` | User turn prefix |
| `<｜Assistant｜>` | Assistant turn prefix |
| `<｜System｜>` | Mid-conversation system message prefix |
| `<｜latest_reminder｜>` | Latest reminder (date, locale, etc.) |
| `<think>` / `</think>` | Reasoning block delimiters |
| `｜DSML｜` | DSML markup token |
| `<｜deepseek_image｜>` | Image placeholder in the prompt string |

### Roles

The encoding supports the following message roles: `system`, `user`, `assistant`,
`tool`, and `latest_reminder`.

A `tool` message is not rendered directly: `merge_tool_messages()` converts it
into a `<tool_result>` block inside the preceding user message. When multiple
tool results are present, they are sorted by the order of the corresponding
`tool_calls` in the preceding assistant message.

### Basic chat

A simple multi-turn conversation is encoded as:

```
<｜begin▁of▁sentence｜>{system_prompt}
<｜User｜>{user_message}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>
<｜User｜>{user_message_2}<｜Assistant｜></think>{response_2}<｜end▁of▁sentence｜>
```

- The BOS token is prepended at the very beginning of the conversation.
- In **chat mode** (`thinking_mode="chat"`), `</think>` is placed right after
  `<｜Assistant｜>` to immediately close the thinking block, so the model generates
  content directly.

### Thinking mode

In **thinking mode** (`thinking_mode="thinking"`), the model produces explicit
reasoning inside `<think>...</think>` blocks before responding.

```
<｜begin▁of▁sentence｜><｜System｜>{reasoning_effort_prefix}{system_prompt}
<｜User｜>{message}<｜Assistant｜><think>{reasoning}</think>{response}<｜end▁of▁sentence｜>
```

The reasoning effort prefix is injected once, before the system message, as a
`<｜System｜>` block:

```
<｜System｜>Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)
```

The `drop_thinking` parameter (default `True`) controls whether reasoning from
earlier turns is preserved:

- **Without a tools declaration**: reasoning before the latest user request is
  stripped. Tool results alone do not start a new user turn, so reasoning within
  the current turn is retained across tool calls.
- **With a tools declaration**: `drop_thinking` is automatically disabled. All
  turns retain their reasoning, because tool-calling conversations require full
  context for the model to track multi-step reasoning across tool calls.

### Tool calling (DSML format)

Tools are defined on the `system` message via the `tools` field
(OpenAI-compatible format). When tools are present, the following schema block is
injected into the system prompt:

```
## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

{tool_definitions_json}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
```

An actual tool call in the assistant turn looks like:

```xml

<｜DSML｜ calls>
<｜DSML｜ invoke name="function_name">
<｜DSML｜ parameter name="param" string="true">string_value</｜DSML｜ parameter>
<｜DSML｜ parameter name="count" string="false">5</｜DSML｜ parameter>
</｜DSML｜ invoke>
</｜DSML｜ calls><｜end▁of▁sentence｜>
```

- `string="true"`: the parameter value is a raw string.
- `string="false"`: the parameter value is JSON (number, boolean, array, object).

Tool execution results are wrapped in `<tool_result>` tags within user messages:

```
<｜User｜><tool_result>{result_json}</tool_result><｜Assistant｜><think>...
```

### Tool namespaces

Tool definitions may include a `namespace` alongside `function`, either as a
string or as an object with `name` and an optional `description`:

```python
tool = {
    "type": "function",
    "namespace": {"name": "search", "description": "Search tools."},
    "function": {
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}
tool_call = {
    "type": "function",
    "namespace": "search",
    "function": {"name": "lookup", "arguments": '{"query": "value"}'},
}
```

The tool schema and DSML invocation both use `search::lookup`. The namespace
description is prepended to the tool description, separated by a newline.
The parser returns `function.name="lookup"` and `namespace="search"` on the
tool call, so its output can be passed back to `encode_messages()` directly.

Input also accepts `namespace` inside `function`, or a qualified function name
such as `search::lookup`. A qualified name must agree with any explicit
namespace; `::` separates exactly one namespace from the tool name. Tools
without a namespace retain their original names and output format.

### Reasoning effort

Pass `reasoning_effort` as an integer in `[1, 100]` or as one of `"low"` (50),
`"high"` (75), or `"max"` (100). The default is `"high"` (75).
The setting only affects `thinking_mode="thinking"` and is only rendered at the
start of the conversation (index 0). Intermediate values may be used to elicit
interpolated reasoning behavior.

### Quick instruction special tokens

Quick instruction tokens are used for auxiliary classification and generation
tasks. They are appended to messages via the `"task"` field to trigger
specialized model behavior for a single-token or short-form output.

| Special Token | Description | Format |
|:---|:---|:---|
| `<｜action｜>` | Determines whether the user prompt requires a web search or can be answered directly. | `...<｜User｜>{prompt}<｜Assistant｜><think><｜action｜>` |
| `<｜title｜>` | Generates a concise conversation title after the first assistant response. | `...<｜Assistant｜>{response}<｜end▁of▁sentence｜><｜title｜>` |
| `<｜query｜>` | Generates search queries for the user prompt. | `...<｜User｜>{prompt}<｜query｜>` |
| `<｜authority｜>` | Classifies the user prompt's demand for source authoritativeness. | `...<｜User｜>{prompt}<｜authority｜>` |
| `<｜domain｜>` | Identifies the domain of the user prompt. | `...<｜User｜>{prompt}<｜domain｜>` |
| `<｜read_url｜>` | Determines whether each URL in the user prompt should be fetched and read. | `...<｜User｜>{prompt}<｜read_url｜>` |

Usage in message format:

- **`action`** on a user message: the `<｜action｜>` token is placed after the
  assistant prefix and thinking token, triggering a routing decision (e.g.,
  "Search" or "Answer").
- **Other tasks** (`query`, `authority`, `domain`, `read_url`) on a user message:
  the task token is appended directly after the user content.
- **`title`** on an assistant message: the `<｜title｜>` token is appended after
  the assistant's EOS. The next assistant message provides the generated title.

## Tests

From this directory:

```bash
python -m pytest -q test_encoding.py
```

Test cases are stored as paired JSON input / TXT expected-output files under
`tests/`. The tests cover multi-turn conversations, tool calling, thinking mode,
numeric reasoning effort, mid-conversation system messages, and multimodal image
ordering. They include a check that the TXT and JSON examples encode to the same
prompt and preserve the same image ordering.
