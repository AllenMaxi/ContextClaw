from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .protocol import LLMResponse, ToolCall


class OllamaProvider:
    """LLM provider backed by a local Ollama instance (no extra dependencies)."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(
        self, messages: list[dict], tools: list[dict], system: str = ""
    ) -> LLMResponse:
        all_messages: list[dict[str, Any]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": all_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url=f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.base_url}. Make sure Ollama is running (ollama serve)."
            ) from e

        message = raw.get("message", {})
        content_text: str = message.get("content") or ""
        tool_calls: list[ToolCall] = []

        for tc in message.get("tool_calls") or []:
            func = tc.get("function", {})
            arguments = func.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"_raw": arguments}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=arguments,
                )
            )

        usage: dict[str, int] = {}
        if "prompt_eval_count" in raw:
            usage["input_tokens"] = raw["prompt_eval_count"]
        if "eval_count" in raw:
            usage["output_tokens"] = raw["eval_count"]

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage,
        )
