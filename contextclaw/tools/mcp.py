from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _format_mcp_content(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list) and content:
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=True))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=True))
        text = "\n".join(part for part in parts if part)
        if text:
            return text
    return json.dumps(result, ensure_ascii=True)


@dataclass
class MCPServerConfig:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""


class MCPServerClient:
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        request_timeout: float = 20.0,
    ) -> None:
        self.config = config
        self.request_timeout = request_timeout
        self.process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._io_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.process is not None:
            return

        env = os.environ.copy()
        env.update(self.config.env)

        self.process = await asyncio.create_subprocess_exec(
            *self.config.command,
            cwd=self.config.cwd or None,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._initialize()

    async def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5.0)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()
        self.process = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def list_tools(self) -> list[dict[str, Any]]:
        response = await self._send_request("tools/list", {})
        tools = response.get("tools", [])
        return tools if isinstance(tools, list) else []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        response = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        return _format_mcp_content(response)

    async def _initialize(self) -> None:
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ContextClaw", "version": "0.1.0"},
            },
        )
        await self._send_notification("notifications/initialized", {})

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        await self._write_message(payload)

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError(f"MCP server '{self.config.name}' is not running")

        async with self._io_lock:
            self._request_id += 1
            request_id = self._request_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            await self._write_message(payload)

            while True:
                message = await asyncio.wait_for(
                    self._read_message(),
                    timeout=self.request_timeout,
                )
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise RuntimeError(
                        f"MCP server '{self.config.name}' error for {method}: {message['error']}"
                    )
                result = message.get("result", {})
                return result if isinstance(result, dict) else {"result": result}

    async def _write_message(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f"MCP server '{self.config.name}' is not running")
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8") + b"\n"
        self.process.stdin.write(data)
        await self.process.stdin.drain()

    async def _read_message(self) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError(f"MCP server '{self.config.name}' is not running")
        while True:
            raw = await self.process.stdout.readline()
            if not raw:
                raise RuntimeError(f"MCP server '{self.config.name}' closed its stdout")
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON MCP stdout from %s: %s", self.config.name, line)
                continue
            if isinstance(message, dict):
                return message

    async def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        while True:
            raw = await self.process.stderr.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                logger.debug("[MCP:%s] %s", self.config.name, line)


def load_mcp_registry_config(path: Path, resolve_env) -> list[MCPServerConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data.get("servers", [])
    if not isinstance(servers, list):
        raise ValueError("MCP registry must contain a 'servers' list")

    configs: list[MCPServerConfig] = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        command = entry.get("command", [])
        if not name or not isinstance(command, list) or not command:
            continue
        env_raw = entry.get("env", {})
        env: dict[str, str] = {}
        if isinstance(env_raw, dict):
            for key, value in env_raw.items():
                env[str(key)] = resolve_env(str(value))
        cwd = str(entry.get("cwd", "")).strip()
        configs.append(
            MCPServerConfig(
                name=name,
                command=[str(part) for part in command],
                env=env,
                cwd=cwd,
            )
        )
    return configs
