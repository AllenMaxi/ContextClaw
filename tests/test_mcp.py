"""Tests for MCP registry loading and tool execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from contextclaw.config.agent_config import AgentConfig
from contextclaw.providers.protocol import LLMResponse, ToolCall
from contextclaw.runtime import create_tools
from contextclaw.runner import AgentRunner
from contextclaw.tools.manager import ToolManager


class FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._queue = list(responses)

    def complete(self, messages: list, tools: list, system: str = "") -> LLMResponse:
        if self._queue:
            return self._queue.pop(0)
        return LLMResponse(content="done")


def _server_script(tmp_path: Path) -> Path:
    script = tmp_path / "mock_mcp_server.py"
    script.write_text(
        """
import json
import sys

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "mock", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            },
        }
        print(json.dumps(response), flush=True)
    elif method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text back",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            },
        }
        print(json.dumps(response), flush=True)
    elif method == "tools/call":
        text = message["params"]["arguments"]["text"]
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {"content": [{"type": "text", "text": text}]},
        }
        print(json.dumps(response), flush=True)
""",
        encoding="utf-8",
    )
    return script


def _registry_file(tmp_path: Path, script: Path) -> Path:
    registry = tmp_path / "mcp_servers.json"
    registry.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "demo",
                        "command": [sys.executable, str(script)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return registry


@pytest.mark.asyncio
async def test_tool_manager_loads_mcp_registry_and_calls_tool(tmp_path: Path):
    script = _server_script(tmp_path)
    registry = _registry_file(tmp_path, script)

    manager = ToolManager()
    await manager.load_mcp_registry(registry)

    try:
        assert manager.get_tool("mcp__demo__echo") is not None
        result = await manager.call_mcp_tool("mcp__demo__echo", {"text": "hello"})
        assert result == "hello"
    finally:
        await manager.stop_all()


@pytest.mark.asyncio
async def test_runner_executes_mcp_tool(tmp_path: Path):
    script = _server_script(tmp_path)
    registry = _registry_file(tmp_path, script)

    manager = ToolManager()
    await manager.load_mcp_registry(registry)
    try:
        tc = ToolCall(id="tc1", name="mcp__demo__echo", arguments={"text": "from mcp"})
        provider = FakeProvider(
            [
                LLMResponse(content="", tool_calls=[tc]),
                LLMResponse(content="Done"),
            ]
        )
        config = AgentConfig(name="mcp-agent", workspace=tmp_path)
        runner = AgentRunner(
            config=config, provider=provider, tools=manager, min_call_interval=0
        )

        events = []
        async for event in runner.run("Use MCP"):
            events.append(event)

        result = next(e for e in events if e.type == "tool_result")
        assert result.data["result"] == "from mcp"
    finally:
        await manager.stop_all()


@pytest.mark.asyncio
async def test_create_tools_merges_manual_and_generated_registries(tmp_path: Path):
    script = _server_script(tmp_path)
    manual_registry = _registry_file(tmp_path, script)
    generated_registry = tmp_path / ".contextclaw" / "generated" / "mcp_servers.json"
    generated_registry.parent.mkdir(parents=True)
    generated_registry.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "demo",
                        "command": [sys.executable, str(script)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / ".contextclaw" / "catalog.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "version": 1,
                "connectors": [
                    {
                        "id": "planning-bundle",
                        "bundles": ["planning"],
                        "missing_prerequisites": [],
                    }
                ],
                "skills": [],
                "generated": {
                    "mcp_servers_path": str(generated_registry),
                    "policy_path": "",
                },
            }
        ),
        encoding="utf-8",
    )

    config = AgentConfig(
        name="catalog-agent",
        workspace=tmp_path,
        mcp_servers_path=manual_registry,
        tools=["filesystem"],
    )
    manager = await create_tools(config)
    try:
        names = {tool["name"] for tool in manager.list_tools()}
        assert "filesystem_read" in names
        assert "write_todos" in names
        assert "mcp__demo__echo" in names
    finally:
        await manager.stop_all()
