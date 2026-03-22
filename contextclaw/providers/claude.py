from __future__ import annotations

import os
from typing import Any

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "anthropic SDK is required for ClaudeProvider. Install it with: pip install anthropic"
    ) from e

from .protocol import LLMProvider, LLMResponse, ToolCall


class ClaudeProvider:
    """LLM provider backed by Anthropic's Claude API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def complete(
        self, messages: list[dict], tools: list[dict], system: str = ""
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=4096,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tools:
            # Anthropic API expects 'input_schema' not 'parameters'
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("input_schema", t.get("parameters", {})),
                }
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)

        content_text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage,
        )


# Satisfy the Protocol at import time (structural subtyping check)
_: LLMProvider = ClaudeProvider.__new__(ClaudeProvider)  # type: ignore[assignment]
