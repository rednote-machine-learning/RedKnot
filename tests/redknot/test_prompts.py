import unittest

from benchmarks.redknot.prompts import prompt_protocol


class PromptProtocolTests(unittest.TestCase):
    def test_deepseek_single_user_chat_protocol(self) -> None:
        render, provenance = prompt_protocol("deepseek-v4.1-flash", object())
        result = render("Question: Who wrote Hamlet?\nAnswer:")
        self.assertEqual(
            result,
            "<｜begin▁of▁sentence｜><｜User｜>Question: Who wrote Hamlet?\nAnswer:<｜Assistant｜></think>",
        )
        self.assertEqual(provenance["thinking_mode"], "chat")
        self.assertEqual(len(provenance["renderer_sha256"]), 64)

    def test_other_models_require_their_own_template(self) -> None:
        with self.assertRaisesRegex(ValueError, "chat template"):
            prompt_protocol("qwen3-32b", object())

    def test_use_selected_models_chat_template(self) -> None:
        class Tokenizer:
            chat_template = "model-specific"

            def apply_chat_template(self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool) -> str:
                self.call = messages, tokenize, add_generation_prompt
                return "template:" + messages[0]["content"]

        tokenizer = Tokenizer()
        render, _ = prompt_protocol("qwen3-32b", tokenizer)
        self.assertEqual(render("question"), "template:question")
        self.assertEqual(tokenizer.call, ([{"role": "user", "content": "question"}], False, True))


if __name__ == "__main__":
    unittest.main()
