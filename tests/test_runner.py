"""Tests for AgentRunner with a fake provider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from contextclaw.config.agent_config import AgentConfig
from contextclaw.providers.protocol import LLMResponse, ToolCall
from contextclaw.runner import AgentRunner, Event
from contextclaw.tools.manager import ToolDefinition, ToolManager

# ---------------------------------------------------------------------------
# FakeProvider
# ---------------------------------------------------------------------------


class FakeProvider:
    """Returns canned LLMResponse objects from a queue."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._queue = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self._queue:
            return self._queue.pop(0)
        return LLMResponse(content="(no more responses)")


class FailingProvider:
    """Provider that raises on every call."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.call_count = 0

    def complete(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        self.call_count += 1
        raise self._exc


class TransientThenOkProvider:
    """Fails N times with a transient error, then succeeds."""

    def __init__(self, fail_count: int, success_response: LLMResponse) -> None:
        self._fail_count = fail_count
        self._response = success_response
        self.call_count = 0

    def complete(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise ConnectionError("transient failure")
        return self._response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **kwargs) -> AgentConfig:
    defaults = dict(name="test-agent", workspace=tmp_path)
    defaults.update(kwargs)
    return AgentConfig(**defaults)


async def _collect(runner: AgentRunner, message: str) -> list[Event]:
    events = []
    async for event in runner.run(message):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Simple run — no tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_run_yields_text_and_done(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Hello from agent!")])
    runner = AgentRunner(config=config, provider=provider)

    events = await _collect(runner, "Hi")

    types = [e.type for e in events]
    assert "text" in types
    assert "done" in types

    text_event = next(e for e in events if e.type == "text")
    assert text_event.data["content"] == "Hello from agent!"

    done_event = next(e for e in events if e.type == "done")
    assert done_event.data["content"] == "Hello from agent!"


@pytest.mark.asyncio
async def test_simple_run_no_tool_call_events(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Answer")])
    runner = AgentRunner(config=config, provider=provider)

    events = await _collect(runner, "What is 2+2?")
    types = [e.type for e in events]
    assert "tool_call" not in types
    assert "tool_result" not in types


# ---------------------------------------------------------------------------
# Run with tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_tool_call_yields_tool_events(tmp_path: Path):
    tc = ToolCall(id="tc1", name="shell_execute", arguments={"command": "echo hi"})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Done"),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    runner = AgentRunner(config=config, provider=provider)

    events = await _collect(runner, "Run echo")

    types = [e.type for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert "done" in types


@pytest.mark.asyncio
async def test_run_tool_call_event_data(tmp_path: Path):
    tc = ToolCall(id="tc42", name="web_search", arguments={"query": "pytest"})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Found results"),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    # Register the tool so validation passes
    tools = ToolManager()
    tools.register(ToolDefinition(name="web_search", description="Search"))
    runner = AgentRunner(config=config, provider=provider, tools=tools)

    events = await _collect(runner, "Search")
    call_events = [e for e in events if e.type == "tool_call"]
    assert len(call_events) == 1
    assert call_events[0].data["id"] == "tc42"
    assert call_events[0].data["name"] == "web_search"
    assert call_events[0].data["arguments"] == {"query": "pytest"}


# ---------------------------------------------------------------------------
# Policy blocking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_blocking_yields_tool_result_with_blocked_message(tmp_path: Path):
    tc = ToolCall(id="tc_blocked", name="shell_execute", arguments={"command": "rm -rf /"})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="I cannot execute that"),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)

    policy = MagicMock()
    policy.check_tool.return_value = "block"

    runner = AgentRunner(config=config, provider=provider, policy=policy)
    events = await _collect(runner, "Delete everything")

    result_events = [e for e in events if e.type == "tool_result"]
    assert len(result_events) == 1
    assert "blocked by policy" in result_events[0].data["result"]


@pytest.mark.asyncio
async def test_policy_allow_does_not_block(tmp_path: Path):
    tc = ToolCall(id="tc_ok", name="filesystem_read", arguments={"path": "/tmp/file.txt"})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="File content retrieved"),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register(ToolDefinition(name="filesystem_read", description="Read file"))

    policy = MagicMock()
    policy.check_tool.return_value = "allow"

    runner = AgentRunner(config=config, provider=provider, policy=policy, tools=tools)
    events = await _collect(runner, "Read file")

    # Tool should have been executed (stub result), not blocked
    result_events = [e for e in events if e.type == "tool_result"]
    assert len(result_events) == 1
    assert "blocked by policy" not in result_events[0].data["result"]


# ---------------------------------------------------------------------------
# Knowledge recall integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_recall_emits_event_when_memories_exist(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Answer using memory")])

    knowledge = MagicMock()
    knowledge.recall.return_value = [{"content": "previous fact"}]
    knowledge.store.return_value = None

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)
    events = await _collect(runner, "What do you know?")

    recall_events = [e for e in events if e.type == "knowledge_recalled"]
    assert len(recall_events) == 1
    assert "memories" in recall_events[0].data
    assert recall_events[0].data["memories"][0]["content"] == "previous fact"


@pytest.mark.asyncio
async def test_knowledge_recall_no_event_when_empty(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Fresh answer")])

    knowledge = MagicMock()
    knowledge.recall.return_value = []
    knowledge.store.return_value = None

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)
    events = await _collect(runner, "Tell me something")

    recall_events = [e for e in events if e.type == "knowledge_recalled"]
    assert recall_events == []


@pytest.mark.asyncio
async def test_knowledge_store_called_on_done(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Significant output")])

    knowledge = MagicMock()
    knowledge.recall.return_value = []
    knowledge.store.return_value = None

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)
    await _collect(runner, "Do something")

    knowledge.store.assert_called_once_with("Significant output")


# ---------------------------------------------------------------------------
# Max turns limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_turns_limit_yields_error_event(tmp_path: Path):
    """When the agent keeps calling tools, it should hit max turns and yield an error."""
    # Always return a tool call — never terminates on its own
    tc = ToolCall(id="tc_inf", name="some_tool", arguments={})
    # Provide enough responses to cover _max_turns iterations
    responses = [LLMResponse(content="", tool_calls=[tc])] * 25
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    runner = AgentRunner(config=config, provider=provider, max_turns=3)

    events = await _collect(runner, "Loop forever")

    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "Max turns" in error_events[0].data["message"]


# ---------------------------------------------------------------------------
# SOUL.md system prompt loading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soul_md_loaded_as_system_prompt(tmp_path: Path):
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text(
        "---\nname: Tester\nrole: tester\n---\nYou are a test agent.\n",
        encoding="utf-8",
    )
    config = _make_config(tmp_path, soul_path=soul_file)
    provider = FakeProvider([LLMResponse(content="Ready")])
    runner = AgentRunner(config=config, provider=provider)

    # Verify system prompt was loaded
    assert runner._system_prompt == "You are a test agent."

    # Verify it is passed to complete()
    await _collect(runner, "Start")
    assert provider.calls[0]["system"] == "You are a test agent."


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


def test_event_creation():
    event = Event(type="text", data={"content": "hello"})
    assert event.type == "text"
    assert event.data["content"] == "hello"


def test_event_default_data():
    event = Event(type="done")
    assert event.data == {}


# ---------------------------------------------------------------------------
# Smart recall — uses context from prior turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smart_recall_uses_context_from_prior_turns(tmp_path: Path):
    """After the first turn, recall query should include last assistant content."""
    config = _make_config(tmp_path)
    provider = FakeProvider(
        [
            LLMResponse(content="First answer"),
            LLMResponse(content="Second answer"),
        ]
    )

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.return_value = None
    knowledge.agent_id = "agent-1"

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)

    # First turn — recall query is just the user message
    await _collect(runner, "Hello")
    first_query = knowledge.recall.call_args_list[0][0][0]
    assert first_query == "Hello"

    # Second turn — recall query should include prior assistant content
    await _collect(runner, "Follow up")
    second_query = knowledge.recall.call_args_list[1][0][0]
    assert "First answer" in second_query
    assert "Follow up" in second_query


@pytest.mark.asyncio
async def test_recall_happens_every_turn(tmp_path: Path):
    """Recall should be called on each run(), not just the first."""
    config = _make_config(tmp_path)
    provider = FakeProvider(
        [
            LLMResponse(content="A1"),
            LLMResponse(content="A2"),
            LLMResponse(content="A3"),
        ]
    )

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.return_value = None
    knowledge.agent_id = "agent-1"

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)

    await _collect(runner, "Q1")
    await _collect(runner, "Q2")
    await _collect(runner, "Q3")

    assert knowledge.recall.call_count == 3


# ---------------------------------------------------------------------------
# close_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_session_calls_summarize_and_store(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider(
        [
            LLMResponse(content="A1"),
            LLMResponse(content="A2"),
        ]
    )

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.return_value = None
    knowledge.agent_id = "agent-1"
    knowledge.summarize_and_store.return_value = [{"content": "learned fact"}]

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)

    await _collect(runner, "Q1")
    await _collect(runner, "Q2")

    stored = await runner.close_session()

    knowledge.summarize_and_store.assert_called_once()
    call_kwargs = knowledge.summarize_and_store.call_args
    assert call_kwargs[1]["provider"] is provider
    assert call_kwargs[1]["agent_name"] == "test-agent"
    assert stored == [{"content": "learned fact"}]


@pytest.mark.asyncio
async def test_close_session_skips_short_sessions(tmp_path: Path):
    """Sessions with fewer than 2 user turns should not be summarized."""
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="A1")])

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.return_value = None
    knowledge.agent_id = "agent-1"

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)

    # Only one turn
    await _collect(runner, "Q1")

    stored = await runner.close_session()
    assert stored == []
    knowledge.summarize_and_store.assert_not_called()


@pytest.mark.asyncio
async def test_close_session_without_knowledge_returns_empty(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="A1")])
    runner = AgentRunner(config=config, provider=provider, knowledge=None)

    await _collect(runner, "Q1")
    stored = await runner.close_session()
    assert stored == []


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_recovers_from_transient_failure(tmp_path: Path):
    """Provider fails once, then succeeds on retry."""
    config = _make_config(tmp_path)
    provider = TransientThenOkProvider(
        fail_count=1,
        success_response=LLMResponse(content="Recovered!"),
    )
    runner = AgentRunner(
        config=config,
        provider=provider,
        retry_base_delay=0.01,  # fast for tests
    )

    events = await _collect(runner, "Hello")
    types = [e.type for e in events]
    assert "text" in types
    assert "done" in types
    text_event = next(e for e in events if e.type == "text")
    assert text_event.data["content"] == "Recovered!"
    assert provider.call_count == 2  # 1 fail + 1 success


@pytest.mark.asyncio
async def test_retry_exhausted_yields_error(tmp_path: Path):
    """Provider fails on all retries — should yield error event."""
    config = _make_config(tmp_path)
    provider = FailingProvider(ConnectionError("down"))
    runner = AgentRunner(
        config=config,
        provider=provider,
        max_retries=2,
        retry_base_delay=0.01,
    )

    events = await _collect(runner, "Hello")
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert "Provider error" in error_events[0].data["message"]
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_non_transient_error_no_retry(tmp_path: Path):
    """ValueError should NOT be retried."""
    config = _make_config(tmp_path)
    provider = FailingProvider(ValueError("bad input"))
    runner = AgentRunner(
        config=config,
        provider=provider,
        max_retries=3,
        retry_base_delay=0.01,
    )

    events = await _collect(runner, "Hello")
    error_events = [e for e in events if e.type == "error"]
    assert len(error_events) == 1
    assert provider.call_count == 1  # No retry


# ---------------------------------------------------------------------------
# Tool validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_tool_rejected_with_error(tmp_path: Path):
    """LLM requests a tool that doesn't exist — should get validation error."""
    tc = ToolCall(id="tc_bad", name="nonexistent_tool", arguments={"x": 1})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="OK"),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    runner = AgentRunner(config=config, provider=provider)

    events = await _collect(runner, "Do something")
    result_events = [e for e in events if e.type == "tool_result"]
    assert len(result_events) == 1
    assert "Unknown tool" in result_events[0].data["result"]


@pytest.mark.asyncio
async def test_shell_execute_always_valid(tmp_path: Path):
    """shell_execute is always valid even without explicit registration."""
    tc = ToolCall(id="tc_sh", name="shell_execute", arguments={"command": "echo hi"})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Done"),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    runner = AgentRunner(config=config, provider=provider)

    events = await _collect(runner, "Run it")
    result_events = [e for e in events if e.type == "tool_result"]
    assert len(result_events) == 1
    # Should NOT contain "Unknown tool"
    assert "Unknown tool" not in result_events[0].data["result"]


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_usage_tracks_tokens(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider(
        [
            LLMResponse(content="A1", usage={"input_tokens": 10, "output_tokens": 20}),
            LLMResponse(content="A2", usage={"input_tokens": 15, "output_tokens": 25}),
        ]
    )
    runner = AgentRunner(config=config, provider=provider)

    await _collect(runner, "Q1")
    await _collect(runner, "Q2")

    assert runner.total_usage["input_tokens"] == 25
    assert runner.total_usage["output_tokens"] == 45


# ---------------------------------------------------------------------------
# Knowledge store failure is non-fatal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_store_failure_does_not_crash(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Output")])

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.side_effect = RuntimeError("store crashed")

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)
    events = await _collect(runner, "Test")

    # Should still complete successfully despite store failure
    types = [e.type for e in events]
    assert "done" in types
    assert "error" not in types


# ---------------------------------------------------------------------------
# close_session handles summarization failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_session_handles_summarization_failure(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider(
        [
            LLMResponse(content="A1"),
            LLMResponse(content="A2"),
        ]
    )

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.return_value = None
    knowledge.agent_id = "agent-1"
    knowledge.summarize_and_store.side_effect = RuntimeError("boom")

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)
    await _collect(runner, "Q1")
    await _collect(runner, "Q2")

    # Should not raise, returns empty
    stored = await runner.close_session()
    assert stored == []
