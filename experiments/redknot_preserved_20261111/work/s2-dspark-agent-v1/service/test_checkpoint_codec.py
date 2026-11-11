import json
import os
from pathlib import Path
import sys
import unittest

from checkpoint_codec import CheckpointCodec, ENCODER_SHA256, safe_prefix

STAGE = Path(__file__).resolve().parent.parent
try:
    from tokenizers import Tokenizer
except ImportError:
    sys.path.insert(0, str(STAGE.parent / "code-agent-redknot-v1/_deps"))
    from tokenizers import Tokenizer


class CheckpointCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(os.environ.get("S2_TOKENIZER",
            str(STAGE.parent / "benchmark_rag/tokenizer/tokenizer.json")))
        cls.codec = CheckpointCodec(STAGE / "checkpoint-encoding/encoding.py", cls.tokenizer)

    def test_actual_checkpoint_multi_turn_tool_roundtrip(self):
        messages = [dict(role="system", content="You fix code."), dict(role="user", content="Read the source."),
            dict(role="assistant", content="", reasoning_content="Need source.", tool_calls=[
                dict(id="read1", type="function", function=dict(name="read_file", arguments='{"path":"a.py"}'))]),
            dict(role="tool", tool_call_id="read1", content="print('hello')\n"*300)]
        tools = [dict(type="function", function=dict(name="read_file", description="Read a file",
            parameters=dict(type="object", properties=dict(path=dict(type="string")))))]
        rendered = self.codec.render(dict(messages=messages, tools=tools))
        self.assertEqual(rendered["encoder_sha256"], ENCODER_SHA256)
        self.assertIn("<tool_result>", rendered["prompt"])
        self.assertIn("Need source.</think>", rendered["prompt"])
        self.assertTrue(rendered["prompt"].endswith("<｜Assistant｜><think>"))
        self.assertEqual(rendered["input_ids"][0], 0)
        doc = next(d for d in rendered["documents"] if d["origin"]["role"] == "tool")
        self.assertGreater(doc["tokens"], 256)
        self.assertEqual(doc["origin"]["tool_call_id"], "read1")

    def test_duplicate_tool_bodies_have_distinct_occurrences_equal_content_ids(self):
        body = "hello world\n" * 300
        payload = dict(messages=[dict(role="user", content="inspect"),
            dict(role="assistant", content="", tool_calls=[
                dict(id=k, type="function", function=dict(name="read_file", arguments="{}")) for k in ("a", "b")]),
            dict(role="tool", tool_call_id="b", content=body),
            dict(role="tool", tool_call_id="a", content=body)])
        docs = [d for d in self.codec.render(payload)["documents"] if d["origin"]["role"] == "tool"]
        self.assertEqual(len(docs), 2)
        self.assertLess(docs[0]["input_start"], docs[1]["input_start"])
        self.assertEqual([d["origin"]["tool_call_id"] for d in docs], ["a", "b"])
        self.assertEqual(docs[0]["document_id"], docs[1]["document_id"])

    def test_separate_user_skill_content_block_maps_without_task_text(self):
        payload = dict(messages=[dict(role="user", content=[dict(type="text", text="Task A"),
            dict(type="text", text="Reusable coding rules.\n"*300)])])
        result = self.codec.render(payload)
        self.assertEqual(len(result["documents"]), 2)
        self.assertLess(result["documents"][0]["tokens"], 256)
        self.assertGreater(result["documents"][1]["tokens"], 256)

    def test_natural_system_user_and_rendered_tools_preserve_exact_provider_prompt(self):
        tools = [dict(type="function", function=dict(name="inspect", description="Read source "*500,
            parameters=dict(type="object", properties=dict(path=dict(type="string")))))]
        messages = [dict(role="system",content="Repository rules.\n"*200),
            dict(role="user",content="A task description.\n"*200)]
        original_fn = self.codec.encoder.render_tools
        original_globals = self.codec.encoder.encode_messages.__globals__
        result = self.codec.render(dict(messages=messages,tools=tools))
        provider_messages = json.loads(json.dumps(messages))
        provider_messages[0]["tools"] = tools
        expected = self.codec.encoder.encode_messages(provider_messages, thinking_mode="thinking", reasoning_effort=75)
        self.assertEqual(result["prompt"], expected)
        self.assertEqual(list(result["input_ids"]), self.tokenizer.encode(expected,add_special_tokens=False).ids)
        self.assertEqual({d["origin"]["source_kind"] for d in result["documents"]},
            {"system_body","user_body","checkpoint_tool_template"})
        self.assertIs(self.codec.encoder.render_tools, original_fn)
        self.assertIs(self.codec.encoder.encode_messages.__globals__, original_globals)
        self.assertEqual(self.codec.render(dict(messages=messages,tools=tools))["documents"], result["documents"])
        self.assertEqual([d["input_start"] for d in result["documents"]],
            sorted(d["input_start"] for d in result["documents"]))

    def test_tool_text_blocks_are_explicit_natural_source_boundaries(self):
        payload = dict(messages=[dict(role="user",content="Read"),
            dict(role="assistant",content="",tool_calls=[dict(id="a",type="function",
                function=dict(name="read_file",arguments="{}"))]),
            dict(role="tool",tool_call_id="a",content=[dict(type="text",text="Source one\n"*300),
                dict(type="text",text="Source two\n"*300)])])
        result = self.codec.render(payload)
        docs = [d for d in result["documents"] if d["origin"]["role"] == "tool"]
        self.assertEqual(len(docs),2)
        self.assertEqual([d["origin"]["content_block_index"] for d in docs],[0,1])
        self.assertTrue(all(d["tokens"] > 256 for d in docs))

    def stream(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        stream = self.codec.completion_stream("r1")
        events = []
        for index, token in enumerate(ids):
            events.extend(stream.feed(token, finished=index == len(ids)-1,
                finish_reason="stop" if index == len(ids)-1 else ""))
        return events

    def test_real_tokenized_unicode_stream_matches_checkpoint_parse(self):
        events = self.stream("先检查代码。</think>答案是正确的。<｜end▁of▁sentence｜>")
        self.assertEqual("".join(e.get("reasoning_content", "") for e in events), "先检查代码。")
        self.assertEqual("".join(e.get("content", "") for e in events), "答案是正确的。")
        self.assertEqual(events[-1], dict(finish_reason="stop"))

    def test_dsml_is_parsed_to_real_tool_call_not_emitted_as_content(self):
        text = ('Check file.</think>I will read it.\n\n<｜DSML｜ calls>\n'
            '<｜DSML｜ invoke name="read_file">\n'
            '<｜DSML｜ parameter name="path" string="true">a.py</｜DSML｜ parameter>\n'
            '</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>')
        events = self.stream(text)
        content = "".join(e.get("content", "") for e in events)
        self.assertEqual(content, "I will read it.")
        calls = [e["tool_calls"][0] for e in events if "tool_calls" in e]
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), dict(path="a.py"))
        self.assertEqual(events[-1], dict(finish_reason="tool_calls"))

    def test_special_delimiter_partial_is_held(self):
        self.assertEqual(safe_prefix("hello\n\n<｜DS", ["\n\n<｜DSML｜ calls"]), "hello")

    def test_unsupported_protocol_is_rejected_not_dropped(self):
        for change in (dict(stop=["END"]), dict(tool_choice="required"),
                       dict(messages=[dict(role="user", content=[dict(type="image_url", image_url="x")])])):
            with self.assertRaises(ValueError):
                self.codec.render({"messages": [dict(role="user", content="hi")], **change})


if __name__ == "__main__":
    unittest.main()
