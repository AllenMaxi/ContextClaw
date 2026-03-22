from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


class ToolManager:
    """Manages tool definitions and MCP server processes."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._mcp_clients: dict[str, Any] = {}
        self._mcp_tool_bindings: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool

    def register_bundle(
        self, bundle_name: str, bundles_path: Path | None = None
    ) -> None:
        """Load and register all tools from a named bundle.

        Args:
            bundle_name: Name of the bundle (e.g. "filesystem", "web", "shell").
            bundles_path: Optional override path to the bundles JSON file.
        """
        from .bundles import load_bundle

        for tool in load_bundle(bundle_name, bundles_path):
            self.register(tool)

    # ------------------------------------------------------------------
    # MCP server lifecycle
    # ------------------------------------------------------------------

    async def start_mcp_server(self, name: str, command: list[str]) -> None:
        """Start an MCP server subprocess.

        Args:
            name: Logical name used to track this server.
            command: The command and arguments to launch the server.

        Raises:
            ValueError: If a server with this name is already running.
        """
        if name in self._processes:
            raise ValueError(f"MCP server '{name}' is already running.")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes[name] = process

    async def stop_mcp_server(self, name: str) -> None:
        """Stop a named MCP server subprocess.

        Args:
            name: The logical name used when the server was started.

        Raises:
            KeyError: If no server with this name is running.
        """
        if name not in self._processes:
            raise KeyError(f"No running MCP server named '{name}'.")

        process = self._processes.pop(name)
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except TimeoutError:
            process.kill()
            await process.wait()

    async def stop_all(self) -> None:
        """Stop all running MCP server subprocesses."""
        names = list(self._processes.keys())
        for name in names:
            await self.stop_mcp_server(name)
        client_names = list(self._mcp_clients.keys())
        for name in client_names:
            client = self._mcp_clients.pop(name)
            await client.stop()

    async def load_mcp_registry(self, path: Path) -> None:
        """Start MCP servers from a registry file and register their tools."""
        from ..config.agent_config import _resolve_env
        from .mcp import MCPServerClient, load_mcp_registry_config

        for server in load_mcp_registry_config(path, _resolve_env):
            client = MCPServerClient(server)
            await client.start()
            self._mcp_clients[server.name] = client
            remote_tools = await client.list_tools()
            for tool in remote_tools:
                remote_name = str(tool.get("name", "")).strip()
                if not remote_name:
                    continue
                local_name = f"mcp__{server.name}__{remote_name}"
                params = tool.get("inputSchema", tool.get("parameters", {}))
                description = str(tool.get("description", "")).strip()
                prefix = f"[MCP:{server.name}] "
                self.register(
                    ToolDefinition(
                        name=local_name,
                        description=prefix + description if description else prefix + remote_name,
                        parameters=params if isinstance(params, dict) else {},
                    )
                )
                self._mcp_tool_bindings[local_name] = (server.name, remote_name)

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """Return all registered tools as dicts (suitable for an LLM provider)."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Return a tool by name, or None if not registered."""
        return self._tools.get(name)

    def is_mcp_tool(self, name: str) -> bool:
        return name in self._mcp_tool_bindings

    async def call_mcp_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._mcp_tool_bindings:
            raise KeyError(f"Unknown MCP tool '{name}'")
        server_name, remote_name = self._mcp_tool_bindings[name]
        client = self._mcp_clients.get(server_name)
        if client is None:
            raise RuntimeError(f"MCP server '{server_name}' is not running")
        return await client.call_tool(remote_name, arguments)
