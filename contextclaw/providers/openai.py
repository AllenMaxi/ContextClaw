from __future__ import annotations

import json
import os
from typing import Any

try:
    import openai as openai_sdk
except ImportError as e:
    raise ImportError(
        "openai SDK is required for OpenAIProvider. "
        "Install it with: pip install openai"
    ) from e

from .protocol import LLMProvider, LLMResponse, ToolCall


class OpenAIProvider:
    """LLM provider backed by OpenAI-compatible APIs (OpenAI, Azure, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self._client = openai_sdk.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def complete(
        self, messages: list[dict], tools: list[dict], system: str = ""
    ) -> LLMResponse:
        all_messages: list[dict[str, Any]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=all_messages,
        )
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)

        if not response.choices:
            return LLMResponse(content="", tool_calls=[], usage={})

        message = response.choices[0].message
        content_text = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": tc.function.arguments}
                if not isinstance(arguments, dict):
                    arguments = {"_raw": arguments}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    )
                )

        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage,
        )
