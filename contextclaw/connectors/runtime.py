from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config.agent_config import _resolve_env
from ..tools.mcp import MCPServerClient, MCPServerConfig
from .github import GitHubConnectorClient
from .playwright import PlaywrightConnectorClient

logger = logging.getLogger(__name__)


def _trim_text(text: str, output_limit_tokens: int) -> str:
    if output_limit_tokens <= 0:
        return text
    limit = max(output_limit_tokens * 4, 512)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _default_endpoint(definition: "ConnectorRuntimeDefinition") -> str:
    runtime = definition.runtime
    if runtime.url:
        return runtime.url
    if runtime.command:
        return " ".join(runtime.command_parts)
    return runtime.adapter or runtime.transport


@dataclass
class ConnectorRuntimeSpec:
    driver: str
    transport: str
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    headers_env: dict[str, str] = field(default_factory=dict)
    auth: str = "none"
    capabilities: list[str] = field(default_factory=lambda: ["tools"])
    tool_allowlist: list[str] = field(default_factory=list)
    tool_prefix: str = ""
    default_policy: dict[str, list[str]] = field(default_factory=dict)
    timeouts: dict[str, float] = field(default_factory=dict)
    output_limit_tokens: int = 0
    doctor_checks: list[str] = field(default_factory=list)
    docs_url: str = ""
    adapter: str = ""

    @property
    def command_parts(self) -> list[str]:
        parts: list[str] = []
        if self.command:
            parts.append(self.command)
        parts.extend(self.args)
        return parts


@dataclass
class ConnectorRuntimeDefinition:
    id: str
    display_name: str
    description: str
    stability: str
    tags: list[str] = field(default_factory=list)
    required_env: list[str] = field(default_factory=list)
    missing_env: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    missing_prerequisites: list[str] = field(default_factory=list)
    tools_exposed: list[str] = field(default_factory=list)
    workspace: Path = Path(".")
    runtime: ConnectorRuntimeSpec = field(
        default_factory=lambda: ConnectorRuntimeSpec(
            driver="managed_mcp",
            transport="stdio",
            name="connector",
        )
    )


@dataclass
class ConnectorSnapshot:
    id: str
    display_name: str
    driver: str
    transport: str
    endpoint: str
    authenticated: bool
    healthy: bool
    auth_pending: bool
    tool_count: int
    tool_names: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    missing_env: list[str] = field(default_factory=list)
    missing_prerequisites: list[str] = field(default_factory=list)
    startup_error: str = ""
    docs_url: str = ""


def load_connector_registry_config(path: Path) -> list[ConnectorRuntimeDefinition]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("connectors", [])
    if not isinstance(entries, list):
        raise ValueError("Connector registry must contain a 'connectors' list")

    workspace = (
        path.resolve().parents[2] if len(path.resolve().parents) >= 3 else path.parent
    )
    definitions: list[ConnectorRuntimeDefinition] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        runtime_raw = entry.get("runtime", {})
        if not isinstance(runtime_raw, dict):
            continue
        required_env = [
            str(item).strip()
            for item in entry.get("required_env", [])
            if str(item).strip()
        ]
        missing_env = [
            name for name in required_env if not os.environ.get(name, "").strip()
        ]
        prerequisites = [
            str(item).strip()
            for item in entry.get("prerequisites", [])
            if str(item).strip()
        ]
        missing_prerequisites = [
            command for command in prerequisites if shutil.which(command) is None
        ]
        definitions.append(
            ConnectorRuntimeDefinition(
                id=str(entry.get("id", "")).strip(),
                display_name=str(entry.get("display_name", "")).strip()
                or str(entry.get("id", "")).strip(),
                description=str(entry.get("description", "")).strip(),
                stability=str(entry.get("stability", "")).strip(),
                tags=[
                    str(item).strip()
                    for item in entry.get("tags", [])
                    if str(item).strip()
                ],
                required_env=required_env,
                missing_env=missing_env,
                prerequisites=prerequisites,
                missing_prerequisites=missing_prerequisites,
                tools_exposed=[
                    str(item).strip()
                    for item in entry.get("tools_exposed", [])
                    if str(item).strip()
                ],
                workspace=workspace,
                runtime=ConnectorRuntimeSpec(
                    driver=str(runtime_raw.get("driver", "managed_mcp")).strip(),
                    transport=str(runtime_raw.get("transport", "stdio")).strip(),
                    name=str(runtime_raw.get("name", entry.get("id", ""))).strip()
                    or str(entry.get("id", "")).strip(),
                    command=str(runtime_raw.get("command", "")).strip(),
                    args=[
                        str(item).strip()
                        for item in runtime_raw.get("args", [])
                        if str(item).strip()
                    ],
                    url=str(runtime_raw.get("url", "")).strip(),
                    cwd=str(runtime_raw.get("cwd", "")).strip(),
                    env={
                        str(key): _resolve_env(str(value))
                        for key, value in runtime_raw.get("env", {}).items()
                    }
                    if isinstance(runtime_raw.get("env", {}), dict)
                    else {},
                    headers_env={
                        str(key): str(value)
                        for key, value in runtime_raw.get("headers_env", {}).items()
                    }
                    if isinstance(runtime_raw.get("headers_env", {}), dict)
                    else {},
                    auth=str(runtime_raw.get("auth", "none")).strip() or "none",
                    capabilities=[
                        str(item).strip()
                        for item in runtime_raw.get("capabilities", ["tools"])
                        if str(item).strip()
                    ]
                    or ["tools"],
                    tool_allowlist=[
                        str(item).strip()
                        for item in runtime_raw.get("tool_allowlist", [])
                        if str(item).strip()
                    ],
                    tool_prefix=str(runtime_raw.get("tool_prefix", "")).strip()
                    or f"mcp__{str(entry.get('id', '')).strip()}",
                    default_policy={
                        "require_confirm": [
                            str(item).strip()
                            for item in runtime_raw.get("default_policy", {}).get(
                                "require_confirm", []
                            )
                            if str(item).strip()
                        ],
                        "blocked": [
                            str(item).strip()
                            for item in runtime_raw.get("default_policy", {}).get(
                                "blocked", []
                            )
                            if str(item).strip()
                        ],
                    }
                    if isinstance(runtime_raw.get("default_policy", {}), dict)
                    else {"require_confirm": [], "blocked": []},
                    timeouts={
                        str(key): float(value)
                        for key, value in runtime_raw.get("timeouts", {}).items()
                    }
                    if isinstance(runtime_raw.get("timeouts", {}), dict)
                    else {},
                    output_limit_tokens=int(
                        runtime_raw.get("output_limit_tokens", 0) or 0
                    ),
                    doctor_checks=[
                        str(item).strip()
                        for item in runtime_raw.get(
                            "doctor_checks",
                            ["env", "prerequisites", "connectivity", "tools"],
                        )
                        if str(item).strip()
                    ],
                    docs_url=str(runtime_raw.get("docs_url", "")).strip(),
                    adapter=str(runtime_raw.get("adapter", "")).strip(),
                ),
            )
        )
    return definitions


class HTTPMCPClient:
    def __init__(
        self,
        *,
        name: str,
        url: str,
        headers_env: dict[str, str],
        request_timeout: float = 20.0,
    ) -> None:
        self.name = name
        self.url = url
        self.headers_env = headers_env
        self.request_timeout = request_timeout
        self._request_id = 0

    async def start(self) -> None:
        await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ContextClaw", "version": "0.1.0"},
            },
        )

    async def stop(self) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._send_request("tools/list", {})
        tools = result.get("tools", [])
        return tools if isinstance(tools, list) else []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            if parts:
                return "\n".join(parts)
        return json.dumps(result, ensure_ascii=True)

    async def _send_request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self._request_id += 1
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            },
            ensure_ascii=True,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        for header, env_ref in self.headers_env.items():
            resolved = _resolve_env(env_ref) or os.environ.get(env_ref, "")
            if resolved:
                headers[header] = resolved
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            response_text = await asyncio.to_thread(self._do_request, request)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MCP HTTP endpoint unreachable: {exc.reason}") from exc
        response = json.loads(response_text)
        if "error" in response:
            raise RuntimeError(
                f"MCP server '{self.name}' error for {method}: {response['error']}"
            )
        result = response.get("result", {})
        return result if isinstance(result, dict) else {"result": result}

    def _do_request(self, request: urllib.request.Request) -> str:
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            return response.read().decode("utf-8", errors="replace")


class ManagedMCPConnectorClient:
    def __init__(self, definition: ConnectorRuntimeDefinition) -> None:
        self.definition = definition
        self.runtime = definition.runtime
        self.client: MCPServerClient | HTTPMCPClient | None = None
        self._tool_cache: list[dict[str, Any]] = []
        self.startup_error = ""

    async def start(self) -> None:
        if self.runtime.transport == "sse":
            raise RuntimeError("SSE transport is not yet supported by ContextClaw.")
        if self.runtime.transport == "stdio":
            config = MCPServerConfig(
                name=self.runtime.name,
                command=self.runtime.command_parts,
                env=self.runtime.env,
                cwd=self.runtime.cwd,
            )
            timeout = self.runtime.timeouts.get("request", 20.0)
            self.client = MCPServerClient(config, request_timeout=timeout)
        else:
            timeout = self.runtime.timeouts.get("request", 20.0)
            self.client = HTTPMCPClient(
                name=self.runtime.name,
                url=self.runtime.url,
                headers_env=self.runtime.headers_env,
                request_timeout=timeout,
            )
        await self.client.start()

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.stop()
            self.client = None

    async def list_tools(self) -> list[dict[str, Any]]:
        if self.client is None:
            return []
        self._tool_cache = await self.client.list_tools()
        return self._tool_cache

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self.client is None:
            raise RuntimeError("Connector client is not running")
        result = await self.client.call_tool(name, arguments)
        return _trim_text(result, self.runtime.output_limit_tokens)

    async def health(self) -> dict[str, Any]:
        if self.client is None:
            return {"healthy": False, "authenticated": not self.definition.missing_env}
        try:
            tools = await self.list_tools()
        except Exception as exc:  # noqa: BLE001
            return {
                "healthy": False,
                "authenticated": not self.definition.missing_env,
                "message": str(exc),
            }
        return {
            "healthy": True,
            "authenticated": not self.definition.missing_env,
            "tool_count": len(tools),
        }


class PythonAdapterConnectorClient:
    def __init__(self, definition: ConnectorRuntimeDefinition) -> None:
        self.definition = definition
        self.runtime = definition.runtime
        self._client = self._build_client()
        self.startup_error = ""

    def _build_client(self) -> Any:
        if self.runtime.adapter == "github":
            token = os.environ.get("GITHUB_TOKEN", "")
            timeout = self.runtime.timeouts.get("request", 20.0)
            return GitHubConnectorClient(
                token=token,
                base_url=self.runtime.url or "https://api.github.com",
                request_timeout=timeout,
                output_limit_tokens=self.runtime.output_limit_tokens,
            )
        if self.runtime.adapter == "playwright":
            return PlaywrightConnectorClient(
                workspace=self.definition.workspace,
                output_limit_tokens=self.runtime.output_limit_tokens,
            )
        raise ValueError(f"Unknown python adapter '{self.runtime.adapter}'")

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            maybe_awaitable = close()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._client.tool_definitions())

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._client.call_tool(name, arguments)
        return _trim_text(result, self.runtime.output_limit_tokens)

    async def health(self) -> dict[str, Any]:
        health = getattr(self._client, "health", None)
        if callable(health):
            return await health()
        return {"healthy": True, "authenticated": not self.definition.missing_env}


class ConnectorSessionManager:
    def __init__(
        self,
        register_tool: Callable[[str, str, dict[str, Any], str], bool],
    ) -> None:
        self._register_tool = register_tool
        self._clients: dict[
            str, ManagedMCPConnectorClient | PythonAdapterConnectorClient
        ] = {}
        self._bindings: dict[str, tuple[str, str]] = {}
        self._snapshots: dict[str, ConnectorSnapshot] = {}

    async def load_registry(
        self,
        path: Path,
        *,
        skip_existing_connectors: bool = False,
    ) -> None:
        for definition in load_connector_registry_config(path):
            if definition.id in self._clients:
                if skip_existing_connectors:
                    logger.warning(
                        "Skipping connector '%s' from %s because it is already loaded",
                        definition.id,
                        path,
                    )
                    continue
                raise ValueError(f"Connector '{definition.id}' is already loaded.")
            await self.start_connector(definition)

    async def start_connector(self, definition: ConnectorRuntimeDefinition) -> None:
        snapshot = ConnectorSnapshot(
            id=definition.id,
            display_name=definition.display_name,
            driver=definition.runtime.driver,
            transport=definition.runtime.transport,
            endpoint=_default_endpoint(definition),
            authenticated=not definition.missing_env,
            healthy=False,
            auth_pending=bool(definition.missing_env),
            tool_count=0,
            capabilities=list(definition.runtime.capabilities),
            missing_env=list(definition.missing_env),
            missing_prerequisites=list(definition.missing_prerequisites),
            docs_url=definition.runtime.docs_url,
        )

        if definition.runtime.driver == "managed_mcp":
            client = ManagedMCPConnectorClient(definition)
        elif definition.runtime.driver == "python_adapter":
            client = PythonAdapterConnectorClient(definition)
        else:
            raise ValueError(
                f"Unknown connector driver '{definition.runtime.driver}' for {definition.id}"
            )
        self._clients[definition.id] = client

        try:
            await client.start()
            remote_tools = await client.list_tools()
            snapshot.healthy = not definition.missing_prerequisites
        except Exception as exc:  # noqa: BLE001
            snapshot.startup_error = str(exc)
            snapshot.healthy = False
            remote_tools = []
            client.startup_error = str(exc)

        allowed = set(definition.runtime.tool_allowlist)
        registered_tools: list[str] = []
        for tool in remote_tools:
            remote_name = str(tool.get("name", "")).strip()
            if not remote_name:
                continue
            if allowed and remote_name not in allowed:
                continue
            local_name = self._local_tool_name(
                definition.runtime.tool_prefix, remote_name
            )
            params = tool.get("inputSchema", tool.get("parameters", {}))
            description = str(tool.get("description", "")).strip() or remote_name
            if self._register_tool(
                local_name,
                f"[{definition.display_name}] {description}",
                params if isinstance(params, dict) else {},
                definition.id,
            ):
                self._bindings[local_name] = (definition.id, remote_name)
                registered_tools.append(local_name)
        snapshot.tool_names = sorted(registered_tools)
        snapshot.tool_count = len(registered_tools)
        if not snapshot.startup_error and definition.runtime.driver == "python_adapter":
            health = await client.health()
            snapshot.authenticated = bool(
                health.get("authenticated", snapshot.authenticated)
            )
            snapshot.auth_pending = not snapshot.authenticated and bool(
                snapshot.missing_env
            )
            snapshot.healthy = bool(health.get("healthy", snapshot.healthy))
            if health.get("message") and not snapshot.startup_error:
                snapshot.startup_error = str(health["message"])
        self._snapshots[definition.id] = snapshot

    async def stop_all(self) -> None:
        for connector_id, client in list(self._clients.items()):
            try:
                await client.stop()
            finally:
                self._clients.pop(connector_id, None)

    def snapshots(self) -> list[ConnectorSnapshot]:
        return [self._snapshots[key] for key in sorted(self._snapshots)]

    def has_tool(self, local_name: str) -> bool:
        return local_name in self._bindings

    async def call_tool(self, local_name: str, arguments: dict[str, Any]) -> str:
        connector_id, remote_name = self._bindings[local_name]
        client = self._clients[connector_id]
        return await client.call_tool(remote_name, arguments)

    def connector_tools(self, connector_id: str | None = None) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for snapshot in self.snapshots():
            if connector_id and snapshot.id != connector_id:
                continue
            result[snapshot.id] = list(snapshot.tool_names)
        return result

    async def connector_health(
        self,
        connector_id: str | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for snapshot in self.snapshots():
            if connector_id and snapshot.id != connector_id:
                continue
            record = {
                "id": snapshot.id,
                "display_name": snapshot.display_name,
                "driver": snapshot.driver,
                "transport": snapshot.transport,
                "endpoint": snapshot.endpoint,
                "authenticated": snapshot.authenticated,
                "healthy": snapshot.healthy,
                "auth_pending": snapshot.auth_pending,
                "tool_count": snapshot.tool_count,
                "missing_env": snapshot.missing_env,
                "missing_prerequisites": snapshot.missing_prerequisites,
                "startup_error": snapshot.startup_error,
                "docs_url": snapshot.docs_url,
            }
            records.append(record)
        return records

    async def connector_doctor(
        self,
        connector_id: str | None = None,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for snapshot in self.snapshots():
            if connector_id and snapshot.id != connector_id:
                continue
            checks: list[str] = []
            if snapshot.missing_env:
                checks.append("Missing env: " + ", ".join(sorted(snapshot.missing_env)))
            if snapshot.missing_prerequisites:
                checks.append(
                    "Missing prerequisites: "
                    + ", ".join(sorted(snapshot.missing_prerequisites))
                )
            if snapshot.startup_error:
                checks.append(f"Startup: {snapshot.startup_error}")
            if snapshot.healthy:
                checks.append("Runtime: healthy")
            elif not snapshot.startup_error:
                checks.append("Runtime: connector loaded but not healthy")
            reports.append(
                {
                    "id": snapshot.id,
                    "display_name": snapshot.display_name,
                    "checks": checks or ["No issues detected."],
                    "docs_url": snapshot.docs_url,
                }
            )
        return reports

    async def connector_auth(self, connector_id: str) -> dict[str, Any]:
        snapshot = self._snapshots.get(connector_id)
        if snapshot is None:
            raise KeyError(f"Unknown connector '{connector_id}'")
        if snapshot.missing_env:
            return {
                "id": connector_id,
                "authenticated": False,
                "message": "Missing required env vars: "
                + ", ".join(sorted(snapshot.missing_env)),
            }
        if snapshot.driver == "managed_mcp" and snapshot.transport == "sse":
            return {
                "id": connector_id,
                "authenticated": False,
                "message": "OAuth/browser auth is not implemented yet for SSE connectors.",
            }
        return {
            "id": connector_id,
            "authenticated": snapshot.authenticated,
            "message": "Connector credentials are present.",
        }

    @staticmethod
    def _local_tool_name(prefix: str, remote_name: str) -> str:
        prefix = prefix.rstrip("_")
        if prefix.endswith("__"):
            return f"{prefix}{remote_name}"
        if prefix:
            return f"{prefix}__{remote_name}"
        return remote_name
