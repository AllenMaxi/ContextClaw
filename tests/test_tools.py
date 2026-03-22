"""Tests for ToolManager and bundle loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from contextclaw.tools.bundles import load_bundle
from contextclaw.tools.manager import ToolDefinition, ToolManager

# ---------------------------------------------------------------------------
# ToolDefinition
# ---------------------------------------------------------------------------


def test_tool_definition_creation():
    tool = ToolDefinition(name="my_tool", description="Does something useful")
    assert tool.name == "my_tool"
    assert tool.description == "Does something useful"
    assert tool.parameters == {}


def test_tool_definition_with_parameters():
    params = {"type": "object", "properties": {"path": {"type": "string"}}}
    tool = ToolDefinition(name="fs_read", description="Read a file", parameters=params)
    assert tool.parameters == params


# ---------------------------------------------------------------------------
# ToolManager — register and get_tool
# ---------------------------------------------------------------------------


def test_register_and_get_tool():
    manager = ToolManager()
    tool = ToolDefinition(name="echo_tool", description="Echoes input")
    manager.register(tool)
    retrieved = manager.get_tool("echo_tool")
    assert retrieved is not None
    assert retrieved.name == "echo_tool"


def test_get_tool_returns_none_for_unknown():
    manager = ToolManager()
    assert manager.get_tool("nonexistent") is None


def test_register_overwrites_existing():
    manager = ToolManager()
    tool_v1 = ToolDefinition(name="my_tool", description="Version 1")
    tool_v2 = ToolDefinition(name="my_tool", description="Version 2")
    manager.register(tool_v1)
    manager.register(tool_v2)
    assert manager.get_tool("my_tool").description == "Version 2"


# ---------------------------------------------------------------------------
# ToolManager — list_tools
# ---------------------------------------------------------------------------


def test_list_tools_empty():
    manager = ToolManager()
    assert manager.list_tools() == []


def test_list_tools_returns_correct_format():
    manager = ToolManager()
    params = {"type": "object", "properties": {"q": {"type": "string"}}}
    manager.register(
        ToolDefinition(name="search", description="Search", parameters=params)
    )
    tools = manager.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "search"
    assert tools[0]["description"] == "Search"
    assert tools[0]["parameters"] == params


def test_list_tools_multiple():
    manager = ToolManager()
    manager.register(ToolDefinition(name="tool_a", description="A"))
    manager.register(ToolDefinition(name="tool_b", description="B"))
    names = {t["name"] for t in manager.list_tools()}
    assert names == {"tool_a", "tool_b"}


# ---------------------------------------------------------------------------
# ToolManager — register_bundle
# ---------------------------------------------------------------------------


def test_register_bundle_filesystem(tmp_path: Path):
    """register_bundle loads the filesystem bundle from the real bundles.json."""
    manager = ToolManager()
    manager.register_bundle("filesystem")
    tools = manager.list_tools()
    names = {t["name"] for t in tools}
    assert "filesystem_read" in names
    assert "filesystem_write" in names
    assert "filesystem_list" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "ls" in names
    assert "edit_file" in names
    assert "glob" in names
    assert "grep" in names


def test_register_bundle_web(tmp_path: Path):
    manager = ToolManager()
    manager.register_bundle("web")
    names = {t["name"] for t in manager.list_tools()}
    assert "web_search" in names
    assert "web_fetch" in names


def test_register_bundle_shell():
    manager = ToolManager()
    manager.register_bundle("shell")
    names = {t["name"] for t in manager.list_tools()}
    assert "shell_execute" in names
    assert "execute" in names


def test_register_bundle_planning():
    manager = ToolManager()
    manager.register_bundle("planning")
    names = {t["name"] for t in manager.list_tools()}
    assert "write_todos" in names
    assert "read_todos" in names


def test_register_bundle_custom(tmp_path: Path):
    """register_bundle accepts a custom bundles_path override."""
    bundle_data = {
        "custom": [
            {"name": "custom_tool", "description": "A custom tool", "parameters": {}}
        ]
    }
    bundles_file = tmp_path / "bundles.json"
    bundles_file.write_text(json.dumps(bundle_data), encoding="utf-8")

    manager = ToolManager()
    manager.register_bundle("custom", bundles_path=bundles_file)
    assert manager.get_tool("custom_tool") is not None


# ---------------------------------------------------------------------------
# load_bundle directly
# ---------------------------------------------------------------------------


def test_load_bundle_valid():
    tools = load_bundle("shell")
    names = {t.name for t in tools}
    assert "shell_execute" in names
    assert "execute" in names


def test_load_bundle_planning():
    tools = load_bundle("planning")
    names = {t.name for t in tools}
    assert "write_todos" in names
    assert "read_todos" in names


def test_load_bundle_invalid_name_raises_key_error():
    with pytest.raises(KeyError, match="does_not_exist"):
        load_bundle("does_not_exist")


def test_load_bundle_missing_file_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_bundle("anything", bundles_path=tmp_path / "nonexistent.json")


def test_load_bundle_custom_path(tmp_path: Path):
    bundle_data = {"mytools": [{"name": "do_thing", "description": "Does a thing"}]}
    bundles_file = tmp_path / "my_bundles.json"
    bundles_file.write_text(json.dumps(bundle_data), encoding="utf-8")
    tools = load_bundle("mytools", bundles_path=bundles_file)
    assert len(tools) == 1
    assert tools[0].name == "do_thing"
    assert tools[0].description == "Does a thing"
    assert tools[0].parameters == {}
