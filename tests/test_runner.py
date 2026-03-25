"""Tests for AgentRunner with a fake provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from contextclaw.config.agent_config import AgentConfig
from contextclaw.chat.session import ChatSession
from contextclaw.providers.protocol import LLMResponse, ToolCall
from contextclaw.runner import AgentRunner, Event
from contextclaw.tools.manager import ToolDefinition, ToolManager
from contextclaw.workflow import WorkflowConfig, write_workflow

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
async def test_simple_run_emits_context_budget_event(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Hello from agent!")])
    runner = AgentRunner(config=config, provider=provider, min_call_interval=0)

    events = await _collect(runner, "Hi")

    budget_event = next(e for e in events if e.type == "context_budget")
    assert budget_event.data["status"] in {"healthy", "warn", "suggest_compact"}
    assert budget_event.data["message_count"] >= 1


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


@pytest.mark.asyncio
async def test_filesystem_read_tool_executes_for_workspace_file(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("workspace note", encoding="utf-8")
    tc = ToolCall(id="tc_fs", name="filesystem_read", arguments={"path": "notes.txt"})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Read complete."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register_bundle("filesystem")
    runner = AgentRunner(
        config=config, provider=provider, tools=tools, min_call_interval=0
    )

    events = await _collect(runner, "Read notes")

    result = next(e for e in events if e.type == "tool_result")
    assert "workspace note" in result.data["result"]


@pytest.mark.asyncio
async def test_filesystem_read_blocks_outside_workspace(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("do not read", encoding="utf-8")
    tc = ToolCall(id="tc_fs", name="filesystem_read", arguments={"path": str(outside)})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Blocked."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register_bundle("filesystem")
    runner = AgentRunner(
        config=config, provider=provider, tools=tools, min_call_interval=0
    )

    events = await _collect(runner, "Read outside")

    result = next(e for e in events if e.type == "tool_result")
    assert "Access denied" in result.data["result"]


@pytest.mark.asyncio
async def test_write_todos_creates_plan_file(tmp_path: Path):
    tc = ToolCall(
        id="tc_plan",
        name="write_todos",
        arguments={"items": ["Inspect repo", "Run tests"], "heading": "Launch Prep"},
    )
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Plan saved."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register_bundle("planning")
    runner = AgentRunner(
        config=config, provider=provider, tools=tools, min_call_interval=0
    )

    events = await _collect(runner, "Create a plan")

    todo_file = tmp_path / ".contextclaw" / "todos.md"
    assert todo_file.exists()
    text = todo_file.read_text(encoding="utf-8")
    assert "# Launch Prep" in text
    assert "- [ ] Inspect repo" in text
    result = next(e for e in events if e.type == "tool_result")
    assert '"item_count": 2' in result.data["result"]


@pytest.mark.asyncio
async def test_read_todos_returns_saved_plan(tmp_path: Path):
    todo_file = tmp_path / ".contextclaw" / "todos.md"
    todo_file.parent.mkdir(parents=True, exist_ok=True)
    todo_file.write_text(
        "# Plan\n\nGenerated by ContextClaw.\n\n- [ ] Record demo\n- [ ] Publish README\n",
        encoding="utf-8",
    )
    tc = ToolCall(id="tc_plan_read", name="read_todos", arguments={})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Plan loaded."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register_bundle("planning")
    runner = AgentRunner(
        config=config, provider=provider, tools=tools, min_call_interval=0
    )

    events = await _collect(runner, "Read the current plan")

    result = next(e for e in events if e.type == "tool_result")
    assert '"exists": true' in result.data["result"].lower()
    assert "Record demo" in result.data["result"]


@pytest.mark.asyncio
async def test_read_file_alias_supports_offset_and_limit(tmp_path: Path):
    target = tmp_path / "notes.txt"
    target.write_text("abcdefghij", encoding="utf-8")
    tc = ToolCall(
        id="tc_alias_read",
        name="read_file",
        arguments={"path": "notes.txt", "offset": 2, "limit": 4},
    )
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Alias read complete."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register_bundle("filesystem")
    runner = AgentRunner(
        config=config, provider=provider, tools=tools, min_call_interval=0
    )

    events = await _collect(runner, "Read a slice")

    result = next(e for e in events if e.type == "tool_result")
    assert '"offset": 2' in result.data["result"]
    assert '"content": "cdef"' in result.data["result"]


@pytest.mark.asyncio
async def test_runner_force_compacts_context_and_reuses_working_memory(
    tmp_path: Path,
):
    workspace = tmp_path / "agents" / "orchestrator"
    workspace.mkdir(parents=True, exist_ok=True)
    write_workflow(
        tmp_path / "Workflow.md",
        WorkflowConfig(
            entry_agent="orchestrator",
            memory_policy={
                "context_window_tokens": 512,
                "reserve_tokens": 64,
                "warn_threshold": 0.20,
                "compact_threshold": 0.30,
                "force_threshold": 0.35,
                "recent_messages_target_tokens": 120,
                "keep_recent_turns": 1,
                "message_preview_chars": 120,
            },
            docs_policy={"mode": "review_queue", "roots": ["docs"]},
        ),
        body="# Workflow\n",
    )
    checkpoint_path = workspace / ".contextclaw" / "session.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    session = ChatSession(max_history=0)
    for index in range(4):
        session.add_user(f"Historical request {index}: " + ("x" * 260))
        session.add_assistant(f"Historical answer {index}: " + ("y" * 220))
    checkpoint_path.write_text(
        json.dumps({"session": session.to_dict(), "total_usage": {}}, indent=2),
        encoding="utf-8",
    )
    config = _make_config(
        tmp_path,
        name="orchestrator",
        workspace=workspace,
        checkpoint_path=checkpoint_path,
    )
    provider = FakeProvider([LLMResponse(content="Compacted answer")])
    knowledge = MagicMock()
    knowledge.auto_recall = False
    knowledge.store.return_value = {"memory": {"memory_id": "mem_compact"}}
    runner = AgentRunner(
        config=config,
        provider=provider,
        knowledge=knowledge,
        min_call_interval=0,
    )

    events = await _collect(runner, "Please continue with the latest task.")

    event_types = [event.type for event in events]
    assert "compact_preview" in event_types
    assert "context_compacted" in event_types
    assert (workspace / ".contextclaw" / "working_memory.json").exists()
    assert "## Working Memory" in provider.calls[0]["system"]
    context_event = next(event for event in events if event.type == "context_compacted")
    assert context_event.data["stored_memory_id"] == "mem_compact"
    compaction_call = knowledge.store.call_args_list[0]
    assert "## Working Memory" in compaction_call.args[0]
    assert compaction_call.kwargs["metadata"] == {
        "source": "context_compaction",
        "agent": "orchestrator",
        "reason": context_event.data["reason"],
    }
    assert compaction_call.kwargs["evidence"] == [context_event.data["artifact_path"]]
    assert compaction_call.kwargs["citations"] == [context_event.data["artifact_path"]]
    assert compaction_call.kwargs["memory_kind"] == "compact"
    assert compaction_call.kwargs["tags"] == [
        "compact",
        "working-memory",
        "orchestrator",
    ]
    assert isinstance(compaction_call.kwargs["section_schema"], dict)
    assert compaction_call.kwargs["importance_score"] == 0.7


@pytest.mark.asyncio
async def test_edit_file_glob_and_grep_tools_work_together(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    target = src / "launch.txt"
    target.write_text("ContextClaw launch draft\n", encoding="utf-8")
    (src / "notes.md").write_text(
        "ContextGraph complements ContextClaw\n", encoding="utf-8"
    )
    responses = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="tc_edit",
                    name="edit_file",
                    arguments={
                        "path": "src/launch.txt",
                        "old_text": "draft",
                        "new_text": "script",
                    },
                ),
                ToolCall(
                    id="tc_glob",
                    name="glob",
                    arguments={"path": "src", "pattern": "*.txt"},
                ),
                ToolCall(
                    id="tc_grep",
                    name="grep",
                    arguments={"path": "src", "pattern": "Context", "include": "*.txt"},
                ),
            ],
        ),
        LLMResponse(content="Workspace search complete."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register_bundle("filesystem")
    runner = AgentRunner(
        config=config, provider=provider, tools=tools, min_call_interval=0
    )

    events = await _collect(runner, "Edit and search files")

    results = [e.data["result"] for e in events if e.type == "tool_result"]
    assert target.read_text(encoding="utf-8") == "ContextClaw launch script\n"
    assert any('"replacements": 1' in result for result in results)
    assert any('"path": "src/launch.txt"' in result for result in results)
    assert any('"line_number": 1' in result for result in results)


# ---------------------------------------------------------------------------
# Policy blocking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_blocking_yields_tool_result_with_blocked_message(tmp_path: Path):
    tc = ToolCall(
        id="tc_blocked", name="shell_execute", arguments={"command": "rm -rf /"}
    )
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
async def test_execute_alias_uses_shell_policy_name(tmp_path: Path):
    tc = ToolCall(id="tc_execute", name="execute", arguments={"command": "pwd"})
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Blocked."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    policy = MagicMock()
    policy.check_tool.side_effect = lambda name: (
        "block" if name == "shell_execute" else "allow"
    )
    tools = ToolManager()
    tools.register_bundle("shell")

    runner = AgentRunner(
        config=config,
        provider=provider,
        tools=tools,
        policy=policy,
        min_call_interval=0,
    )
    events = await _collect(runner, "Run a command")

    result = next(e for e in events if e.type == "tool_result")
    assert "blocked by policy" in result.data["result"]
    assert any(
        call.args[0] == "shell_execute" for call in policy.check_tool.call_args_list
    )


@pytest.mark.asyncio
async def test_policy_allow_does_not_block(tmp_path: Path):
    tc = ToolCall(
        id="tc_ok", name="filesystem_read", arguments={"path": "/tmp/file.txt"}
    )
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


@pytest.mark.asyncio
async def test_policy_confirm_without_approver_blocks_execution(tmp_path: Path):
    tc = ToolCall(
        id="tc_confirm",
        name="filesystem_write",
        arguments={"path": "out.txt", "content": "hello"},
    )
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="I could not write without approval."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register(ToolDefinition(name="filesystem_write", description="Write file"))

    policy = MagicMock()
    policy.check_tool.return_value = "confirm"

    runner = AgentRunner(config=config, provider=provider, policy=policy, tools=tools)
    events = await _collect(runner, "Write a file")

    result_events = [e for e in events if e.type == "tool_result"]
    assert len(result_events) == 1
    assert "requires approval" in result_events[0].data["result"]


@pytest.mark.asyncio
async def test_policy_confirm_with_approver_allows_execution(tmp_path: Path):
    tc = ToolCall(
        id="tc_confirm",
        name="filesystem_write",
        arguments={"path": "out.txt", "content": "hello"},
    )
    responses = [
        LLMResponse(content="", tool_calls=[tc]),
        LLMResponse(content="Done."),
    ]
    config = _make_config(tmp_path)
    provider = FakeProvider(responses)
    tools = ToolManager()
    tools.register(ToolDefinition(name="filesystem_write", description="Write file"))

    policy = MagicMock()
    policy.check_tool.return_value = "confirm"

    async def approver(_tool_call: ToolCall) -> bool:
        return True

    runner = AgentRunner(
        config=config,
        provider=provider,
        policy=policy,
        tools=tools,
        tool_approver=approver,
    )
    events = await _collect(runner, "Write a file")

    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hello"
    result_events = [e for e in events if e.type == "tool_result"]
    assert "Wrote" in result_events[0].data["result"]


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
async def test_durable_memory_recall_is_layered_into_system_prompt(tmp_path: Path):
    config = _make_config(tmp_path)
    provider = FakeProvider([LLMResponse(content="Answer using durable memory")])

    knowledge = MagicMock()
    knowledge.auto_recall = True
    knowledge.recall_memories.return_value = [
        {
            "memory": {
                "memory_id": "mem_1",
                "memory_kind": "compact",
                "content": "Keep the rollout on blue-green deployment.",
                "summary": "Blue-green rollout",
                "tags": ["deploy", "decision"],
            },
            "score": 0.88,
            "matching_claims": [],
        }
    ]
    knowledge.store.return_value = None

    runner = AgentRunner(config=config, provider=provider, knowledge=knowledge)
    events = await _collect(runner, "How should we deploy this safely?")

    recall_event = next(e for e in events if e.type == "knowledge_recalled")
    assert recall_event.data["mode"] == "durable"
    assert "## Durable Memory Recall" in provider.calls[0]["system"]
    assert "Blue-green rollout" in provider.calls[0]["system"]
    assert "[Recalled knowledge]" not in runner.session.last_user_message
    knowledge.recall_memories.assert_called_once()


@pytest.mark.asyncio
async def test_runner_loads_project_and_agent_memory_files(tmp_path: Path):
    project_root = tmp_path / "project"
    workspace = project_root / "agents" / "orchestrator"
    workspace.mkdir(parents=True, exist_ok=True)
    write_workflow(
        project_root / "Workflow.md", WorkflowConfig(entry_agent="orchestrator")
    )
    (project_root / "AGENTS.md").write_text(
        "# Project Memory\n\nAlways use the review-first workflow.\n",
        encoding="utf-8",
    )
    (workspace / "MEMORY.md").write_text(
        "# Agent Memory\n\nThis agent prefers concise release notes.\n",
        encoding="utf-8",
    )
    config = _make_config(
        tmp_path,
        name="orchestrator",
        workspace=workspace,
    )
    provider = FakeProvider([LLMResponse(content="Ready.")])
    runner = AgentRunner(config=config, provider=provider, min_call_interval=0)

    await _collect(runner, "Summarize the changes.")

    assert "## Always-Loaded Memory Files" in provider.calls[0]["system"]
    assert "Always use the review-first workflow." in provider.calls[0]["system"]
    assert "This agent prefers concise release notes." in provider.calls[0]["system"]


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


@pytest.mark.asyncio
async def test_skills_are_appended_to_system_prompt(tmp_path: Path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "planning.md").write_text(
        "Always write a plan first.",
        encoding="utf-8",
    )

    config = _make_config(tmp_path, skills_path=skills_dir)
    provider = FakeProvider([LLMResponse(content="Done")])
    runner = AgentRunner(config=config, provider=provider)

    await _collect(runner, "Start")

    assert "Always write a plan first." in provider.calls[0]["system"]


@pytest.mark.asyncio
async def test_task_tool_delegates_to_subagent(tmp_path: Path):
    subagents_dir = tmp_path / "subagents"
    child_dir = subagents_dir / "research-sub"
    child_dir.mkdir(parents=True)
    (child_dir / "config.yaml").write_text(
        "name: research-sub\nprovider: openai\nsandbox_type: process\ntools: planning\n",
        encoding="utf-8",
    )
    (child_dir / "SOUL.md").write_text(
        "---\nname: research-sub\nrole: research\n---\nDelegated research specialist.\n",
        encoding="utf-8",
    )

    parent_tc = ToolCall(
        id="tc_task",
        name="task",
        arguments={
            "subagent": "research-sub",
            "prompt": "Summarize the delegated work.",
        },
    )
    parent_provider = FakeProvider(
        [
            LLMResponse(content="", tool_calls=[parent_tc]),
            LLMResponse(content="Delegation complete."),
        ]
    )
    child_provider = FakeProvider([LLMResponse(content="Subagent result ready.")])

    def provider_factory(config: AgentConfig):
        if config.name == "research-sub":
            return child_provider
        raise AssertionError(f"Unexpected provider request for {config.name}")

    config = _make_config(tmp_path, subagents_path=subagents_dir)
    tools = ToolManager()
    runner = AgentRunner(
        config=config,
        provider=parent_provider,
        tools=tools,
        provider_factory=provider_factory,
        min_call_interval=0,
    )

    events = await _collect(runner, "Delegate this.")

    result = next(e for e in events if e.type == "tool_result")
    assert "research-sub" in result.data["result"]
    assert "Subagent result ready." in result.data["result"]


@pytest.mark.asyncio
async def test_runner_loads_existing_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / ".contextclaw" / "session.json"
    config = _make_config(tmp_path, checkpoint_path=checkpoint_path)
    first_provider = FakeProvider([LLMResponse(content="First answer")])
    runner = AgentRunner(config=config, provider=first_provider, min_call_interval=0)

    await _collect(runner, "First question")

    second_provider = FakeProvider([LLMResponse(content="Second answer")])
    resumed = AgentRunner(config=config, provider=second_provider, min_call_interval=0)

    assert resumed.session.turn_count == 1
    assert resumed.session.last_assistant_message == "First answer"


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
