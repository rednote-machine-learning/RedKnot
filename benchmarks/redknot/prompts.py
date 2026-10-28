"""Explicit single-turn text prompt protocols, independent of inference kernels."""

from __future__ import annotations

from typing import Any

from .contract import file_digest

DEEPSEEK_V41_ENCODING = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1/encoding/encoding.py"


def prompt_protocol(profile: str, tokenizer: Any) -> tuple:
    """Return (renderer, provenance); no silent raw-prompt fallback.

    The DeepSeek format below is the official reference encoder's restricted
    single-user, text-only, chat-mode case. It deliberately implements no tools,
    images, conversation history, or thinking mode.
    """
    if profile == "deepseek-v4.1-flash":

        def render(content: str) -> str:
            return "<｜begin▁of▁sentence｜><｜User｜>" + content + "<｜Assistant｜></think>"

        return render, {
            "name": "deepseek-v4.1-single-user-chat-v1",
            "source": DEEPSEEK_V41_ENCODING,
            "thinking_mode": "chat",
            "scope": "single-user-text-only",
            "renderer_sha256": file_digest(__file__),
        }
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{profile} requires its model's chat template; refusing to substitute a raw prompt")

    def render_template(content: str) -> str:
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    return render_template, {
        "name": "target-tokenizer-chat-template-v1",
        "scope": "single-user-text-only",
        "renderer_sha256": file_digest(__file__),
    }
