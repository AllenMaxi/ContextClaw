from __future__ import annotations

import asyncio
from typing import Any

from .config.agent_config import AgentConfig


def create_provider(config: AgentConfig) -> Any:
    if config.provider == "claude":
        from .providers.claude import ClaudeProvider

        return ClaudeProvider(model=config.model or "claude-sonnet-4-20250514")
    if config.provider == "openai":
        from .providers.openai import OpenAIProvider

        return OpenAIProvider(model=config.model or "gpt-4o")
    if config.provider == "ollama":
        from .providers.ollama import OllamaProvider

        return OllamaProvider(model=config.model or "llama3.2")
    raise ValueError(f"Unknown provider: {config.provider}")


def create_sandbox(config: AgentConfig) -> Any:
    if config.sandbox_type == "docker":
        from .sandbox.docker import DockerSandbox

        return DockerSandbox(workspace=config.workspace)
    if config.sandbox_type == "process":
        from .sandbox.process import ProcessSandbox

        return ProcessSandbox(workspace=config.workspace)
    return None


def create_knowledge(config: AgentConfig) -> Any | None:
    if not config.cg_url:
        return None

    from .knowledge import ContextGraphBridge

    return ContextGraphBridge(
        cg_url=config.cg_url,
        api_key=config.cg_api_key,
        agent_id=config.agent_id,
    )


def create_policy(config: AgentConfig) -> Any | None:
    if not config.policy_path or not config.policy_path.exists():
        return None

    from .sandbox.policy import PolicyEngine

    return PolicyEngine.from_file(config.policy_path)


async def create_tools(config: AgentConfig):
    from .tools import ToolManager

    tools = ToolManager()
    for bundle_name in config.tools:
        tools.register_bundle(bundle_name)
    if config.mcp_servers_path and config.mcp_servers_path.exists():
        await tools.load_mcp_registry(config.mcp_servers_path)
    return tools


def create_tools_sync(config: AgentConfig):
    return asyncio.run(create_tools(config))
