"""Integration tests — end-to-end agent lifecycle.

These tests wire together real components (config, session, tools, sandbox,
runner) using a fake LLM provider, exercising the full pipeline without
hitting external APIs.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextclaw.chat.session import ChatSession
from contextclaw.config.agent_config import AgentConfig
from contextclaw.providers.protocol import LLMResponse, ToolCall
from contextclaw.runner import AgentRunner, Event
from contextclaw.sandbox.process import ProcessSandbox
from contextclaw.sandbox.policy import PolicyEngine
from contextclaw.tools.manager import ToolManager, ToolDefinition


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Provider that returns pre-scripted responses in order."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._queue = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "system": system})
        if self._queue:
            return self._queue.pop(0)
        return LLMResponse(content="(no more scripted responses)")


def _make_config(tmp_path: Path, **overrides) -> AgentConfig:
    defaults = dict(name="integration-agent", workspace=tmp_path)
    defaults.update(overrides)
    return AgentConfig(**defaults)


async def _collect(runner: AgentRunner, message: str) -> list[Event]:
    events = []
    async for event in runner.run(message):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Full agent lifecycle — create, chat, close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_lifecycle_create_chat_close(tmp_path: Path):
    """Simulate the complete lifecycle: init → multi-turn chat → close."""
    config = _make_config(tmp_path)
    sandbox = ProcessSandbox(workspace=tmp_path)
    tools = ToolManager()
    tools.register_bundle("shell")

    provider = ScriptedProvider([
        LLMResponse(content="Hello! How can I help?"),
        LLMResponse(content="The answer is 42."),
    ])

    runner = AgentRunner(
        config=config,
        provider=provider,
        sandbox=sandbox,
        tools=tools,
        min_call_interval=0,  # no rate limiting in tests
    )

    # Turn 1
    events1 = await _collect(runner, "Hi there")
    types1 = [e.type for e in events1]
    assert "text" in types1
    assert "done" in types1
    assert events1[-1].data["content"] == "Hello! How can I help?"

    # Turn 2
    events2 = await _collect(runner, "What is the meaning of life?")
    text_event = next(e for e in events2 if e.type == "text")
    assert text_event.data["content"] == "The answer is 42."

    # Session state
    assert runner.session.turn_count == 2

    # Close session (no knowledge bridge, should return empty)
    stored = await runner.close_session()
    assert stored == []


# ---------------------------------------------------------------------------
# Tool execution through sandbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_execution_through_sandbox(tmp_path: Path):
    """LLM requests shell_execute → sandbox runs command → result fed back."""
    # Create a file the agent can read
    test_file = tmp_path / "data.txt"
    test_file.write_text("important data\n")

    config = _make_config(tmp_path)
    sandbox = ProcessSandbox(workspace=tmp_path)

    tc = ToolCall(id="tc1", name="shell_execute", arguments={"command": f"cat {test_file}"})
    provider = ScriptedProvider([
        LLMResponse(content="Let me read that file.", tool_calls=[tc]),
        LLMResponse(content="The file contains: important data"),
    ])

    runner = AgentRunner(
        config=config, provider=provider, sandbox=sandbox,
        min_call_interval=0,
    )
    events = await _collect(runner, "Read data.txt")

    # Should have tool_call and tool_result events
    tool_results = [e for e in events if e.type == "tool_result"]
    assert len(tool_results) == 1
    assert "important data" in tool_results[0].data["result"]

    # Final response
    done_event = next(e for e in events if e.type == "done")
    assert "important data" in done_event.data["content"]


# ---------------------------------------------------------------------------
# Sandbox blocks dangerous commands
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_blocks_dangerous_command_in_pipeline(tmp_path: Path):
    """When LLM tries to access blocked paths, sandbox denies it."""
    config = _make_config(tmp_path)
    sandbox = ProcessSandbox(workspace=tmp_path)

    tc = ToolCall(id="tc1", name="shell_execute", arguments={"command": "cat ~/.ssh/id_rsa"})
    provider = ScriptedProvider([
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="I can't access that file."),
    ])

    runner = AgentRunner(
        config=config, provider=provider, sandbox=sandbox,
        min_call_interval=0,
    )
    events = await _collect(runner, "Show me SSH keys")

    tool_results = [e for e in events if e.type == "tool_result"]
    assert len(tool_results) == 1
    assert "Access denied" in tool_results[0].data["result"]


# ---------------------------------------------------------------------------
# Policy engine integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_blocks_tool_in_full_pipeline(tmp_path: Path):
    """Policy engine blocks a tool call before it reaches the sandbox."""
    policy_yaml = """\
permissions:
  tools:
    blocked:
      - shell_execute
"""
    config = _make_config(tmp_path)
    policy = PolicyEngine.from_text(policy_yaml)
    sandbox = ProcessSandbox(workspace=tmp_path)

    tc = ToolCall(id="tc1", name="shell_execute", arguments={"command": "echo hi"})
    provider = ScriptedProvider([
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Shell is blocked, I'll try another way."),
    ])

    runner = AgentRunner(
        config=config, provider=provider,
        sandbox=sandbox, policy=policy,
        min_call_interval=0,
    )
    events = await _collect(runner, "Run a command")

    tool_results = [e for e in events if e.type == "tool_result"]
    assert any("blocked by policy" in e.data["result"] for e in tool_results)


# ---------------------------------------------------------------------------
# Knowledge recall + store integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_recall_and_store_full_cycle(tmp_path: Path):
    """Knowledge is recalled before the turn, stored after the turn."""
    config = _make_config(tmp_path)
    provider = ScriptedProvider([
        LLMResponse(content="Based on recalled knowledge, the answer is X."),
    ])

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = [{"content": "User prefers concise answers"}]
    knowledge.store.return_value = {"id": "mem-1"}
    knowledge.agent_id = "agent-test"

    runner = AgentRunner(
        config=config, provider=provider, knowledge=knowledge,
        min_call_interval=0,
    )
    events = await _collect(runner, "Tell me about X")

    # Recall happened
    recall_events = [e for e in events if e.type == "knowledge_recalled"]
    assert len(recall_events) == 1
    assert recall_events[0].data["memories"][0]["content"] == "User prefers concise answers"

    # Store happened
    knowledge.store.assert_called_once()

    # Session messages include recalled context
    messages = runner.session.get_messages()
    recalled_msg = [m for m in messages if "[Recalled knowledge]" in m.get("content", "")]
    assert len(recalled_msg) == 1


# ---------------------------------------------------------------------------
# Multi-turn with tool calls and knowledge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_with_tools_and_knowledge(tmp_path: Path):
    """Complete multi-turn flow: recall → tool call → store → recall again."""
    test_file = tmp_path / "notes.txt"
    test_file.write_text("meeting at 3pm\n")

    config = _make_config(tmp_path)
    sandbox = ProcessSandbox(workspace=tmp_path)

    tc = ToolCall(id="tc1", name="shell_execute", arguments={"command": f"cat {test_file}"})
    provider = ScriptedProvider([
        # Turn 1: LLM reads file
        LLMResponse(content="Let me check your notes.", tool_calls=[tc]),
        LLMResponse(content="You have a meeting at 3pm."),
        # Turn 2: LLM answers directly
        LLMResponse(content="Yes, that's confirmed."),
    ])

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.return_value = None
    knowledge.agent_id = "agent-multi"

    runner = AgentRunner(
        config=config, provider=provider,
        sandbox=sandbox, knowledge=knowledge,
        min_call_interval=0,
    )

    # Turn 1
    events1 = await _collect(runner, "What are my notes?")
    assert any(e.type == "tool_call" for e in events1)
    assert any(e.type == "done" for e in events1)

    # Turn 2
    events2 = await _collect(runner, "Is that right?")
    assert any(e.type == "done" for e in events2)

    # Knowledge recalled on both turns
    assert knowledge.recall.call_count == 2

    # Second recall query should include prior assistant context
    second_query = knowledge.recall.call_args_list[1][0][0]
    assert "meeting at 3pm" in second_query


# ---------------------------------------------------------------------------
# SOUL.md system prompt integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soul_md_used_as_system_prompt(tmp_path: Path):
    """SOUL.md content is passed as system prompt to every provider call."""
    soul_file = tmp_path / "SOUL.md"
    soul_file.write_text(
        "---\nname: TestBot\nrole: assistant\n---\n\nYou are a helpful test bot.\n"
    )

    config = _make_config(tmp_path, soul_path=soul_file)
    provider = ScriptedProvider([
        LLMResponse(content="I am TestBot!"),
    ])

    runner = AgentRunner(
        config=config, provider=provider,
        min_call_interval=0,
    )
    await _collect(runner, "Who are you?")

    # System prompt passed to provider
    assert provider.calls[0]["system"] == "You are a helpful test bot."


# ---------------------------------------------------------------------------
# Session summarization on close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_summarization_on_close(tmp_path: Path):
    """close_session() extracts and stores facts from conversation."""
    config = _make_config(tmp_path)
    provider = ScriptedProvider([
        LLMResponse(content="The API uses REST with JSON payloads."),
        LLMResponse(content="Authentication is via Bearer tokens."),
    ])

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall.return_value = []
    knowledge.store.return_value = None
    knowledge.agent_id = "agent-close"
    knowledge.summarize_and_store.return_value = [
        {"content": "API uses REST + JSON"},
        {"content": "Auth via Bearer tokens"},
    ]

    runner = AgentRunner(
        config=config, provider=provider, knowledge=knowledge,
        min_call_interval=0,
    )

    await _collect(runner, "How does the API work?")
    await _collect(runner, "What about auth?")

    stored = await runner.close_session()

    assert len(stored) == 2
    knowledge.summarize_and_store.assert_called_once()
    # Verify conversation context was passed
    call_kwargs = knowledge.summarize_and_store.call_args
    context = call_kwargs[1].get("conversation_context") or call_kwargs[0][0]
    assert "API" in context


# ---------------------------------------------------------------------------
# Config from directory integration
# ---------------------------------------------------------------------------


def test_agent_config_from_directory(tmp_path: Path):
    """AgentConfig.from_dir wires up workspace, soul, and tools."""
    config_yaml = (
        "name: my-agent\n"
        "provider: claude\n"
        "sandbox_type: process\n"
        "tools: filesystem,shell\n"
    )
    (tmp_path / "config.yaml").write_text(config_yaml)
    (tmp_path / "SOUL.md").write_text(
        "---\nname: my-agent\nrole: default\n---\n\nBe helpful.\n"
    )

    config = AgentConfig.from_dir(tmp_path)

    assert config.name == "my-agent"
    assert config.provider == "claude"
    assert config.sandbox_type == "process"
    assert "filesystem" in config.tools
    assert "shell" in config.tools
    assert config.soul_path is not None
    assert config.soul_path.exists()


# ---------------------------------------------------------------------------
# Rate limiting verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limiting_enforced(tmp_path: Path):
    """Rate limiting adds delay between provider calls."""
    import time

    config = _make_config(tmp_path)
    tc = ToolCall(id="tc1", name="shell_execute", arguments={"command": "echo ok"})
    provider = ScriptedProvider([
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Done"),
    ])
    sandbox = ProcessSandbox(workspace=tmp_path)

    runner = AgentRunner(
        config=config, provider=provider, sandbox=sandbox,
        min_call_interval=0.2,  # 200ms minimum between calls
    )

    start = time.monotonic()
    await _collect(runner, "Go")
    elapsed = time.monotonic() - start

    # Two provider calls with 200ms minimum gap — should take at least ~200ms
    assert elapsed >= 0.15  # small margin for timing jitter
    assert provider.calls  # provider was actually called


# ---------------------------------------------------------------------------
# ChatSession thread safety smoke test
# ---------------------------------------------------------------------------


def test_chat_session_concurrent_access(tmp_path: Path):
    """Multiple threads can read/write the session without errors."""
    import threading

    session = ChatSession(system_prompt="test")
    errors: list[Exception] = []

    def writer(n: int) -> None:
        try:
            for i in range(50):
                session.add_user(f"msg-{n}-{i}")
                session.add_assistant(f"reply-{n}-{i}")
        except Exception as e:
            errors.append(e)

    def reader() -> None:
        try:
            for _ in range(50):
                session.get_messages()
                session.get_summary_context()
                _ = session.turn_count
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(0,)),
        threading.Thread(target=writer, args=(1,)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Thread safety errors: {errors}"
    assert session.turn_count > 0


# ---------------------------------------------------------------------------
# Logging config smoke test
# ---------------------------------------------------------------------------


def test_structured_logging_setup():
    """setup_logging should not raise and should configure the logger."""
    from contextclaw.logging_config import setup_logging
    import logging

    setup_logging(level="DEBUG", structured=True)
    logger = logging.getLogger("contextclaw.test_integration")
    assert logger.getEffectiveLevel() == logging.DEBUG

    # Reset to defaults
    setup_logging(level="WARNING", structured=False)
