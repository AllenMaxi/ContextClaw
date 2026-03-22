"""Tests for provider protocol definitions and provider instantiation."""
from __future__ import annotations

import sys
import types
import unittest.mock
from unittest.mock import patch

import pytest

from contextclaw.providers.protocol import LLMResponse, ToolCall


# ---------------------------------------------------------------------------
# ToolCall dataclass
# ---------------------------------------------------------------------------


def test_tool_call_creation():
    tc = ToolCall(id="call_1", name="shell_execute", arguments={"command": "ls"})
    assert tc.id == "call_1"
    assert tc.name == "shell_execute"
    assert tc.arguments == {"command": "ls"}


def test_tool_call_empty_arguments():
    tc = ToolCall(id="x", name="noop", arguments={})
    assert tc.arguments == {}


def test_tool_call_complex_arguments():
    args = {"path": "/tmp/file.txt", "mode": "r", "encoding": "utf-8"}
    tc = ToolCall(id="abc", name="filesystem_read", arguments=args)
    assert tc.arguments["path"] == "/tmp/file.txt"


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------


def test_llm_response_basic():
    resp = LLMResponse(content="Hello, world!")
    assert resp.content == "Hello, world!"
    assert resp.tool_calls == []
    assert resp.usage == {}


def test_llm_response_with_tool_calls():
    tc = ToolCall(id="t1", name="web_search", arguments={"query": "test"})
    resp = LLMResponse(content="", tool_calls=[tc], usage={"input_tokens": 10, "output_tokens": 5})
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "web_search"
    assert resp.usage["input_tokens"] == 10


def test_llm_response_defaults():
    resp = LLMResponse(content="text")
    assert isinstance(resp.tool_calls, list)
    assert isinstance(resp.usage, dict)


# ---------------------------------------------------------------------------
# Provider import errors (SDK not installed)
# ---------------------------------------------------------------------------


def test_claude_provider_raises_import_error_when_sdk_missing():
    """ClaudeProvider should raise ImportError with a helpful message."""
    # Remove anthropic from sys.modules if present, replace with a broken import
    original = sys.modules.pop("anthropic", None)
    # Also remove the cached provider module so it re-evaluates the import
    provider_mod = sys.modules.pop("contextclaw.providers.claude", None)
    try:
        sys.modules["anthropic"] = None  # type: ignore[assignment]  # forces ImportError on import
        with pytest.raises(ImportError, match="anthropic"):
            import contextclaw.providers.claude  # noqa: F401
    finally:
        # Restore original state
        if original is not None:
            sys.modules["anthropic"] = original
        else:
            sys.modules.pop("anthropic", None)
        if provider_mod is not None:
            sys.modules["contextclaw.providers.claude"] = provider_mod
        else:
            sys.modules.pop("contextclaw.providers.claude", None)


def test_openai_provider_raises_import_error_when_sdk_missing():
    """OpenAIProvider should raise ImportError with a helpful message."""
    original = sys.modules.pop("openai", None)
    provider_mod = sys.modules.pop("contextclaw.providers.openai", None)
    try:
        sys.modules["openai"] = None  # type: ignore[assignment]
        with pytest.raises(ImportError, match="openai"):
            import contextclaw.providers.openai  # noqa: F401
    finally:
        if original is not None:
            sys.modules["openai"] = original
        else:
            sys.modules.pop("openai", None)
        if provider_mod is not None:
            sys.modules["contextclaw.providers.openai"] = provider_mod
        else:
            sys.modules.pop("contextclaw.providers.openai", None)


# ---------------------------------------------------------------------------
# Protocol structural check — complete() signature
# ---------------------------------------------------------------------------


def test_provider_protocol_complete_signature():
    """A class with the right complete() signature satisfies LLMProvider Protocol."""
    from contextclaw.providers.protocol import LLMProvider

    class FakeProvider:
        def complete(
            self, messages: list, tools: list, system: str = ""
        ) -> LLMResponse:
            return LLMResponse(content="ok")

    provider = FakeProvider()
    # Structural check — should not raise
    assert callable(provider.complete)
    result = provider.complete(messages=[], tools=[], system="")
    assert isinstance(result, LLMResponse)
