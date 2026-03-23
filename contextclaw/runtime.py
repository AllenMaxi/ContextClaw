from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .catalog_engine import (
    connector_bundles_from_lock,
    generated_mcp_path,
    generated_policy_path,
)
from .config.agent_config import AgentConfig

logger = logging.getLogger(__name__)


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
    manual_policy = (
        config.policy_path
        if config.policy_path and config.policy_path.exists()
        else None
    )
    generated_policy: Path | None = generated_policy_path(config.workspace)
    if generated_policy is None or not generated_policy.exists():
        generated_policy = None

    if manual_policy is None and generated_policy is None:
        return None

    from .sandbox.policy import PolicyEngine

    return PolicyEngine.from_files(manual_policy, generated_policy)


async def create_tools(config: AgentConfig):
    from .tools import ToolManager

    tools = ToolManager()
    bundle_names: list[str] = []
    for bundle_name in config.tools + connector_bundles_from_lock(config.workspace):
        if bundle_name not in bundle_names:
            bundle_names.append(bundle_name)
    for bundle_name in bundle_names:
        tools.register_bundle(bundle_name)
    if config.mcp_servers_path and config.mcp_servers_path.exists():
        await tools.load_mcp_registry(config.mcp_servers_path)
    generated_mcp = generated_mcp_path(config.workspace)
    if generated_mcp.exists():
        if (
            config.mcp_servers_path
            and generated_mcp.resolve() == config.mcp_servers_path
        ):
            logger.warning(
                "Generated MCP registry matches the manual MCP registry path; skipping duplicate load"
            )
        else:
            await tools.load_mcp_registry(
                generated_mcp,
                skip_existing_servers=True,
            )
    return tools


def create_tools_sync(config: AgentConfig):
    return asyncio.run(create_tools(config))
